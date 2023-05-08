import os
import numpy as np
import matplotlib.pyplot as plt
from NeuralEncoder import NeuralEncoder
from helpers import subjects


def generate_alphas(start, stop, step):
    return [float("{:.8f}".format(10 ** i)) for i in range(start, stop, step)]


def save_kernel_and_score(subject_id, method, kernel, score, folder='kernels'):
    os.makedirs(os.path.join(folder, method), exist_ok=True)
    kernel_filename = os.path.join(folder, method, f'{subject_id}_kernel.npy')
    score_filename = os.path.join(folder, method, f'{subject_id}_score.npy')

    np.save(kernel_filename, kernel)
    np.save(score_filename, score)


def plot_results(encoder, kernel, score):
    times = encoder.model.delays_ / encoder.model.sfreq * 1000

    if encoder.method == 'cortical':
        fig = plt.figure(figsize=(8.0, 4.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        t_min, t_max = -200, 500
        cluster = encoder.cluster

    elif encoder.method == 'subcortical':
        fig = plt.figure(figsize=(8.0, 3.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        t_min, t_max = -5, 20
        cluster = encoder.cluster

    # Kernel
    if encoder.method == 'cortical':
        colors = ['#b2182b', '#ef8a62', '#fddbc7', '#e0e0e0', '#999999', '#4d4d4d']
        for c in range(kernel.shape[0]):
            ax1.plot(times, kernel[c, :], color=colors[c])
        ax1.plot(times, kernel.mean(0), linewidth=15, alpha=0.25, color='gray')
    elif encoder.method == 'subcortical':
        colors = ['#f768a1', '#fcc5c0', '#c51b8a']
        # colors = ['#c51b8a']
        for c in range(kernel.shape[0]):
            ax1.plot(times, kernel[c, :], color=colors[c])

    ax1.set_xlim(t_min, t_max)
    if encoder.method == 'cortical':
        ax1.set_ylim(-0.022, 0.022)
    # if encoder.method == 'subcortical':
    #     ax1.set_ylim(-0.015, 0.015)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Weights')
    ax1.set_title(f'Kernel with best alpha = {encoder.best_alpha}, r = {round(score.mean(0), 4)}')
    ax1.legend(labels=cluster)

    # Hyper-parameter tuning
    x, y = encoder.best_alpha, np.max(encoder.mean_scores_alpha)
    ax2.scatter(x, y, s=100, facecolors='r', edgecolors='r')
    ax2.plot(encoder.alphas, encoder.mean_scores_alpha, '-o', color='blue', label='sin')
    ax2.set_xlabel('Alpha (log-scaled)')
    ax2.set_ylabel('Pearson\'s r')
    ax2.set_xscale('log')
    ax2.set_title('Hyper-parameter tuning')

    if encoder.method == 'cortical':
        # 2D filter
        ax3.imshow(
            kernel,
            cmap='RdBu_r',
            aspect='auto',
            extent=[(times)[0], (times)[-1], 0, len(cluster)],
            # interpolation='spline16'
        )
        ax3.set_xlim(-100, 400)
        ax3.set_title('2D filter')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Channels')
        ax3.set_yticks(np.arange(len(cluster)))
        ax3.set_yticklabels(cluster[::-1])

    plt.tight_layout()
    # plt.show()

    pdf_filename = f'plots/{encoder.method}/{encoder.subject_id}.pdf'
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')

    plt.close()
    del fig


def main():
    alphas = generate_alphas(-3, 13, 2)

    for _, subject_id in enumerate(subjects):
        # print(f'Yay, {subject_id} is encoding!')

        # Cortical encoding
        method = 'cortical'
        encoder = NeuralEncoder(subject_id, method=method, regularization=False, alphas=alphas, init_estimator=100000.0)
        kernel, score = encoder.fit()
        save_kernel_and_score(subject_id, method, kernel, score)

        plot_results(encoder, kernel, score)

        del encoder, kernel, score

        # Subcortical encoding
        method = 'subcortical'
        subcoder = NeuralEncoder(subject_id, method, regularization=False, alphas=alphas, init_estimator=10000000.0)
        kernel, score = subcoder.fit()
        save_kernel_and_score(subject_id, method, kernel, score)

        plot_results(subcoder, kernel, score)

        del subcoder, kernel, score


if __name__ == "__main__":
    main()
