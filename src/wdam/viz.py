import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wdam.utils import ROOT


def finish(name):
    (ROOT / 'results').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(ROOT / 'results' / name, dpi=130)
    plt.close()
