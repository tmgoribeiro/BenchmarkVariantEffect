'''
Code shared by the scripts in scripts/scripts_metrics/: where the outputs of
predict_variant_effects_pooling_strategies.py are, how to read them, the features built from the
ref and alt embeddings, the MLP they train and where they save their results.

The inputs are the outputs of predict_variant_effects_pooling_strategies.py, all in
results/pooling_strategies/<model>/<dataset>/ (see pooling_dir):

    cosine_distance/cosine_distance_<out_file>.csv     distance at the variant position; also the labels
    avg/n{n}/avg_n{n}_<out_file>.csv                   distance of the mean embeddings over [idx - n, idx + n]
    avg/n{n}/embeddings_<out_file>_n{n}.pkl            those mean embeddings: (index, ref, alt) per variant
    ccs/n{n}/ccs_n{n}_<out_file>.csv                   sum of the per-position distances over [idx - n, idx + n]
    cls/cls_<out_file>.csv, cls/embeddings_<out_file>_cls.pkl    [CLS] token (models that have one)
    log.txt                                            AUROC and AUPRC of every strategy, one line per run

Every csv has the bed columns (label, chromosome, ...) and one distance column.

Every output goes to results/<script>/<model>/<dataset>/, or results/<script>/<dataset>/ for the scripts
that only look at a dataset (see results_dir):

    <run>.csv   the results of the run, overwritten when the same run is repeated
    log.txt     one line per run with its settings and main numbers, never overwritten
'''
import argparse
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# bend/utils/<this file> -> repository root (bend is installed in editable mode, so this is the clone)
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'

MODELS = ['ag', 'nt', 'dnabert', 'hyenadna', 'dnabert2', 'grover', 'borzoi']

# how the ref and alt embeddings of a variant are combined into the classifier input
FEATURES = {
    'diff': lambda ref, alt: ref - alt,
    'avg': lambda ref, alt: (ref + alt) / 2,
    'prod': lambda ref, alt: ref * alt,
    'concat': lambda ref, alt: np.concatenate([ref, alt], axis=1),
}


# ---------------------------------------------------------------- paths

def pooling_dir(model, dataset):
    '''Everything predict_variant_effects_pooling_strategies.py writes for one model and dataset.'''
    return RESULTS_DIR / 'pooling_strategies' / model / dataset


def score_folder(score):
    '''Subfolder of pooling_dir of a score: cosine_distance, cls, avg_n8 -> avg/n8, ccs_n8 -> ccs/n8.'''
    if score in ('cosine_distance', 'cls'):
        return Path(score)
    strategy, n = score.split('_')
    return Path(strategy) / n


def score_column(score):
    '''Distance column of a score: cosine_distance -> distance, cls -> distance_cls, avg_n8 -> distance_avg_n8.'''
    return 'distance' if score == 'cosine_distance' else f'distance_{score}'


def scores_path(model, dataset, out_file, score='cosine_distance'):
    '''csv of one score. The cosine_distance csv is also where the labels are read from.'''
    return pooling_dir(model, dataset) / score_folder(score) / f'{score}_{out_file}.csv'


def embeddings_path(model, dataset, out_file, pooling):
    '''pooling n{n}: the mean embeddings over [idx - n, idx + n], in avg/n{n}/; cls: the [CLS] embeddings, in cls/.'''
    if pooling != 'cls' and not re.fullmatch(r'n\d+', pooling):
        raise ValueError(f'pooling must be n<n> or cls, got {pooling!r}')
    folder = score_folder('cls' if pooling == 'cls' else f'avg_{pooling}')
    return pooling_dir(model, dataset) / folder / f'embeddings_{out_file}_{pooling}.pkl'


def dataset_name(bed_file):
    '''variant_effects_disease.bed -> disease. Other names are kept as they are (without the extension).'''
    stem = Path(bed_file).stem
    return stem[len('variant_effects_'):] if stem.startswith('variant_effects_') else stem


def infer_run(path):
    '''
    (model, dataset) of a file written by a script of this repository:

        results/<script>/<model>/<dataset>/...   e.g. results/pooling_strategies/<model>/<dataset>/avg/n8/...
        embeddings/<model>/<run name>/...        the layout of predict_variant_effects.py

    (None, None) for a file anywhere else.
    '''
    parts = Path(path).resolve().parts
    roots = [i for i, part in enumerate(parts[:-1]) if part in ('embeddings', 'results')]
    if not roots:
        return None, None
    i = roots[-1]
    folders = parts[i + 1:-1]  # between embeddings/ or results/ and the file name
    if parts[i] == 'results':
        folders = folders[1:]  # the script folder
    return (folders[0], folders[1]) if len(folders) >= 2 else (None, None)


def resolve_run(parser, path, model=None, dataset=None):
    '''Model and dataset of a results file: --model / --dataset if given, otherwise read from its path (see infer_run).'''
    inferred_model, inferred_dataset = infer_run(path)
    model, dataset = model or inferred_model, dataset or inferred_dataset
    if model is None or dataset is None:
        parser.error(f'Cannot tell the model and dataset of {path} from its path, give --model and --dataset')
    return model, dataset


def results_dir(script_file, models, dataset):
    '''
    results/<script>/<model>/<dataset>/, created if needed. <script> is the script file name without .py;
    for an ensemble, <model> is the combined models joined with +, e.g. dnabert2+grover+hyenadna+ag.
    With models=None (scripts that only look at a dataset): results/<script>/<dataset>/.
    '''
    d = RESULTS_DIR / Path(script_file).stem
    if models is not None:
        d = d / ('+'.join(models) if isinstance(models, (list, tuple)) else models)
    d = d / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(out_dir, run_name, table, log_fields):
    '''Write the metrics table to <run_name>.csv and append the summary line(s) to log.txt.'''
    path = out_dir / f'{run_name}.csv'
    table.to_csv(path, index=False)
    log_fields = [{'run': run_name, **fields} for fields in log_fields]
    for fields in log_fields:
        append_log(out_dir / 'log.txt', fields)
    print(f'\nResults saved to {path} (summary appended to {out_dir / "log.txt"})')


def score_name(value):
    '''argparse type for a score name, as printed by the pooling script: cosine_distance, cls, avg_n{n} or ccs_n{n}.'''
    if value in ('cosine_distance', 'cls') or re.fullmatch(r'(avg|ccs)_n\d+', value):
        return value
    raise argparse.ArgumentTypeError(f'{value!r} is not cosine_distance, cls, avg_n<n> or ccs_n<n>')


def check_files_exist(parser, *paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        parser.error('File not found: ' + ', '.join(missing))


# ---------------------------------------------------------------- arguments

def add_single_model_args(parser, default_features):
    '''Arguments of the scripts that use the embeddings of one model (classifiers/).'''
    parser.add_argument('model', choices=MODELS, type=str, help='Model that computed the embeddings')
    parser.add_argument('dataset', type=str, help='Dataset folder, e.g. disease or expression')
    parser.add_argument('out_file', type=str, help='Run name given to the pooling script')
    add_embedding_args(parser, default_features)
    parser.add_argument('--embeddings', type=str, default=None, help='Embeddings pickle to use instead of the one in results/pooling_strategies/')
    parser.add_argument('--labels', type=str, default=None, help='csv with the labels to use instead of the cosine_distance csv in results/pooling_strategies/')


def add_ensemble_args(parser, default_models):
    '''Arguments of the scripts that combine several models (ensemble/).'''
    parser.add_argument('dataset', type=str, help='Dataset folder, e.g. disease or expression')
    parser.add_argument('--models', choices=MODELS, nargs='+', default=default_models, help='Models to combine')
    parser.add_argument('--out_files', type=str, nargs='+', default=['{dataset}_{model}'],
                        help='Run names given to the pooling script: one per model, in the order of --models, or a single '
                             'template where {model} and {dataset} are replaced, e.g. tgc2048_{model}. Default {dataset}_{model}')


def add_embedding_args(parser, default_features):
    parser.add_argument('--pooling', type=str, default='n0', help='Which embeddings file: n{n} (mean over [idx - n, idx + n]) or cls. Default n0, the variant position alone')
    parser.add_argument('--features', choices=list(FEATURES), nargs='+', default=default_features,
                        help='How ref and alt are combined: diff = ref - alt, avg = (ref + alt) / 2, prod = ref * alt, '
                             'concat = [ref, alt]. Several values run once each')


def add_mlp_args(parser):
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=256, help='0 = linear classifier')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random', action='store_true', help='Skip training: randomly initialised model, as a baseline')


def resolve_out_files(parser, args):
    '''One run name per model, from --out_files (either one per model or a single template).'''
    if len(args.out_files) == 1:
        return [args.out_files[0].format(model=m, dataset=args.dataset) for m in args.models]
    if len(args.out_files) != len(args.models):
        parser.error(f'--out_files needs 1 value or one per model ({len(args.models)}), got {len(args.out_files)}')
    return args.out_files


def single_model_paths(parser, args):
    '''Embeddings and labels files of the classifier scripts, checked to exist.'''
    emb = Path(args.embeddings) if args.embeddings else embeddings_path(args.model, args.dataset, args.out_file, args.pooling)
    labels = Path(args.labels) if args.labels else scores_path(args.model, args.dataset, args.out_file)
    check_files_exist(parser, emb, labels)
    return emb, labels


# ---------------------------------------------------------------- loading

def read_table(path):
    '''Bed files are tab-separated; the csv files of the prediction scripts are comma-separated with the index first.'''
    if Path(path).suffix == '.csv':
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path, sep='\t')


def find_consequence_column(df, column=None):
    '''Name of the consequence column (Consequence in BEND, consequence in TraitGym), or None if there is none.'''
    if column is not None:
        return column if column in df.columns else None
    return next((c for c in df.columns if c.lower() == 'consequence'), None)


def load_pairs(path):
    '''Read the (index, ref, alt) tuples that the prediction scripts pickle one after the other, sorted by index.'''
    idx, ref, alt = [], [], []
    with open(path, 'rb') as f:
        while True:
            try:
                i, r, a = pickle.load(f)
            except EOFError:
                break
            idx.append(i); ref.append(r); alt.append(a)
    idx, ref, alt = np.array(idx), np.stack(ref), np.stack(alt)
    order = np.argsort(idx, kind='stable')
    return idx[order], ref[order], alt[order]


def load_annotation(path, idx):
    '''Rows of the csv (label, chromosome, ...) of the given variants, matched by index rather than by row order.'''
    df = pd.read_csv(path, index_col=0)
    missing = np.setdiff1d(idx, df.index)
    if len(missing) > 0:
        raise ValueError(f'{len(missing)} variant indices are not in {path}, e.g. {missing[:5].tolist()}')
    annotation = df.loc[idx]
    labels = set(annotation['label'].unique())
    if labels != {0, 1}:
        raise ValueError(f'Expected binary labels 0/1 in {path}, got {sorted(labels)}')
    return annotation


def build_features(ref, alt, feature):
    return FEATURES[feature](ref, alt).astype(np.float32)


def load_ensemble_embeddings(parser, args):
    '''ref and alt embeddings of every model in --models, on the same variants in the same order, and their labels.'''
    out_files = resolve_out_files(parser, args)
    emb_paths = [embeddings_path(m, args.dataset, o, args.pooling) for m, o in zip(args.models, out_files)]
    labels_path = scores_path(args.models[0], args.dataset, out_files[0])
    check_files_exist(parser, *emb_paths, labels_path)

    pairs, idx = {}, None
    for model, path in zip(args.models, emb_paths):
        model_idx, ref, alt = load_pairs(path)
        if idx is not None and not np.array_equal(idx, model_idx):
            raise ValueError(f'{model} has embeddings for other variants than {args.models[0]} ({path})')
        idx = model_idx
        pairs[model] = (ref, alt)
        print(f'{model:9s} {path} | {ref.shape}')
    y = load_annotation(labels_path, idx)['label'].to_numpy()
    return pairs, y, out_files


def load_ensemble_scores(parser, args):
    '''One column per model with its --score distance, on the same variants, and their labels.'''
    out_files = resolve_out_files(parser, args)
    column = score_column(args.score)
    paths = [scores_path(m, args.dataset, o, args.score) for m, o in zip(args.models, out_files)]
    check_files_exist(parser, *paths)

    scores, labels = {}, None
    for model, path in zip(args.models, paths):
        df = pd.read_csv(path, index_col=0)
        if column not in df.columns:
            parser.error(f'No column {column} in {path}')
        if labels is not None and not df.index.sort_values().equals(labels.index):
            raise ValueError(f'{model} has distances for other variants than {args.models[0]} ({path})')
        df = df.sort_index()
        labels = df['label'] if labels is None else labels
        scores[model] = df[column].astype(float)
        print(f'{model:9s} {path} | {column}')
    return pd.DataFrame(scores), labels.to_numpy(), out_files


# ---------------------------------------------------------------- MLP

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def to_tensor(X, device):
    return torch.tensor(X, dtype=torch.float32, device=device)


def make_model(d_in, hidden, layers, dropout):
    '''MLP with `layers` hidden layers of size `hidden`, or a linear classifier if either is 0. One output logit.'''
    if hidden == 0 or layers == 0:
        return nn.Linear(d_in, 1)
    modules, d = [], d_in
    for _ in range(layers):
        modules += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    modules.append(nn.Linear(d, 1))
    return nn.Sequential(*modules)


def train_mlp(X_tr, y_tr, args, desc=None):
    '''Train an MLP on tensors already on the device. Prints the loss of every epoch when desc is given.'''
    model = make_model(X_tr.shape[1], args.hidden, args.layers, args.dropout).to(X_tr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    if args.random:
        return model
    n = len(X_tr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=X_tr.device)
        running, steps = torch.zeros((), device=X_tr.device), 0
        for i in range(0, n, args.batch_size):
            batch = perm[i:i + args.batch_size]
            loss = loss_fn(model(X_tr[batch]).squeeze(1), y_tr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.detach()
            steps += 1
        if desc is not None:
            print(f'{desc} | epoch {epoch:3d} | train loss = {running.item() / steps:.4f}')
    return model


def predict_logits(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze(1)


def bce(logits, y):
    '''Validation loss from the logits of the held-out fold.'''
    return nn.functional.binary_cross_entropy_with_logits(logits, y).item()


def out_of_fold_logits(feats, y, args, device):
    '''
    Train one MLP per model in every fold and keep its logits on the held-out variants, so that every
    variant gets one logit per model from an MLP that never saw it. Returns an array (n_variants, n_models).
    '''
    logits = np.zeros((len(y), len(feats)))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        y_tr, y_te = to_tensor(y[tr], device), to_tensor(y[te], device)
        for m, (name, Xm) in enumerate(feats.items()):
            scaler = StandardScaler().fit(Xm[tr])
            X_tr, X_te = to_tensor(scaler.transform(Xm[tr]), device), to_tensor(scaler.transform(Xm[te]), device)
            model = train_mlp(X_tr, y_tr, args)
            fold_logits = predict_logits(model, X_te)
            logits[te, m] = fold_logits.cpu().numpy()
            print(f'Model {name}, fold {fold}: AUC = {roc_auc_score(y[te], logits[te, m]):.4f} | '
                  f'AUPRC = {average_precision_score(y[te], logits[te, m]):.4f} | val loss = {bce(fold_logits, y_te):.4f}')
    return logits


# ---------------------------------------------------------------- reporting

def mlp_settings(args):
    '''Hyperparameters of the MLP, for the log.'''
    return {k: getattr(args, k) for k in ('folds', 'epochs', 'lr', 'weight_decay', 'batch_size', 'hidden', 'layers',
                                          'dropout', 'seed', 'random')}


def summary_fields(rows, y):
    '''Mean +/- std of the AUROC and AUPRC of a list of per-fold rows, for the log.'''
    return {'AUC': mean_std([r['AUC'] for r in rows]).replace(' ', ''),
            'AUPRC': mean_std([r['AUPRC'] for r in rows]).replace(' ', ''),
            'chance_AUPRC': f'{y.mean():.4f}', 'n_variants': len(y)}


def mean_std(values):
    return f'{np.mean(values):.4f} +/- {np.std(values):.4f}'


def fold_std(y, s, folds=5, seed=42):
    '''Std of the AUROC and AUPRC of score s over the stratified folds used by the classifiers, for scores without training.'''
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    m = np.array([(roc_auc_score(y[te], s[te]), average_precision_score(y[te], s[te]))
                  for _, te in skf.split(np.zeros(len(y)), y)])
    return m[:, 0].std(), m[:, 1].std()


def print_summary(title, summary):
    '''summary: {row name: {column: value}}, e.g. one row per feature.'''
    print(f'\n=== {title}')
    print(pd.DataFrame.from_dict(summary, orient='index').to_string())


def now():
    # the GPU runs asynchronously, so wait for it before reading the clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def peak_ram_gb():
    try:
        import resource  # not available on Windows
    except ImportError:
        return float('nan')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def peak_gpu_gb(device):
    return torch.cuda.max_memory_allocated() / 1024**3 if device == 'cuda' else float('nan')


def append_log(path, fields):
    '''One line per run: date | key=value | ...'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(f'{datetime.now():%Y-%m-%d %H:%M:%S} | ' + ' | '.join(f'{k}={v}' for k, v in fields.items()) + '\n')
