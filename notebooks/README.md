# Decentralized Intelligence: From Statistical Foundations to Test-Time Continual Learning

> **A Pedagogical & Experimental Curriculum for Federated Machine Learning**
>
> *Organized sequentially by cognitive complexity: from foundational distributed data
> mechanics to test-time neural memory models.*

______________________________________________________________________

## Pedagogical Structure: The Five Volumes

```
Volume I: Foundations of Federated Learning & Distributional Skew (Beginner)
  ├── 00-flwr-datasets-and-partitioners.ipynb   (Chapter 1)
  ├── 01-centralized-vs-federated.ipynb         (Chapter 2)
  ├── 02-iid-vs-noniid.ipynb                    (Chapter 3 — Research Gap 1)
  └── 03-dynamic-aggregation-strategies.ipynb   (Chapter 4 — Research Gap 2)

Volume II: Cryptographic Privacy Preservation in Distributed Systems (Intermediate)
  ├── 00-differential-privacy-foundations.ipynb (Chapter 5)
  ├── 01-homomorphic-encryption-paillier.ipynb   (Chapter 6)
  ├── 02-secure-multi-party-computation.ipynb   (Chapter 7)
  └── 03-hybrid-privacy-preserving-fl-benchmark.ipynb (Chapter 8)

Volume III: Advanced Geometric Optimization & Matrix Aggregation (Advanced)
  ├── 01-fedmuon-bias-corrected-lmo.ipynb       (Chapter 9)
  └── 02-fedm3-vs-baselines.ipynb               (Chapter 10)

Volume IV: Continual Learning & Mitigating Catastrophic Forgetting (Advanced to Frontier)
  ├── 01-imagenet-transfer-learning-class-incremental.ipynb (Chapter 11)
  └── 02-test-time-training-class-incremental.ipynb         (Chapter 12 — Research Gap 3)

Volume V: Frontier Federated Foundation Models & Dynamic Neural Memory (Capstone)
  └── 01-titans-ttt-centralized-vs-federated.ipynb          (Chapter 13)
```

______________________________________________________________________

## Detailed Chapter Syllabus & Literature Grounding

### Volume I: Foundations of Federated Learning & Distributional Skew

*Target Competency: Understand how distributed clients communicate, how non-IID
partitions diverge, and how standard aggregation operates.*

- **Chapter 1: Decentralized Ingestion & Non-IID Partitioning**
  (`00-preliminaries/00-flwr-datasets-and-partitioners.ipynb`)

  - **Focus**: Setting up decentralized client partitions using Flower Datasets
    (`flwr_datasets`). Explores Uniform IID vs Dirichlet Non-IID label skew
    ($\\text{Dir}(\\alpha)$) and Pathological 2-class allocation.
  - **Literature Citation**:
    - McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
      *Communication-Efficient Learning of Deep Networks from Decentralized Data*. In
      Proceedings of the 20th International Conference on Artificial Intelligence and
      Statistics (AISTATS), PMLR 54:1273-1282.
    - Hsu, T. M. H., Qi, H., & Brown, M. (2019). *Measuring the Effects of Non-Identical
      Distributions on Federated Visual Classification*. arXiv preprint
      arXiv:1909.06335.

- **Chapter 2: Centralized vs. Federated Optimization & The Federation Penalty**
  (`00-preliminaries/01-centralized-vs-federated.ipynb`)

  - **Focus**: Centralized Stochastic Gradient Descent (SGD) versus Federated Averaging
    (FedAvg). Quantifies the "Federation Penalty" (the communication cost, convergence
    lag, and loss discrepancies resulting from decentralized client silos).
  - **Literature Citation**:
    - McMahan et al. (2017), *op. cit.*
    - Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). *Federated Learning:
      Challenges, Methods, and Future Directions*. IEEE Signal Processing Magazine,
      37(3), 50-60.

- **Chapter 3: Statistical Heterogeneity & The Client Drift Phenomenon (Gap 1)**
  (`00-preliminaries/02-iid-vs-noniid.ipynb`)

  - **Focus**: Investigates client drift under non-identical data distributions
    ($\\alpha=0.5$). Demonstrates local minima divergence where local optimizers pull
    client parameters away from the global stationary point. Evaluated on digit
    classification with Register No. 2490031 across MNIST and USPS.
  - **Literature Citation**:
    - Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T.
      (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. In
      International Conference on Machine Learning (ICML), PMLR 119:5132-5143.
    - Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). *Federated
      Learning with Non-IID Data*. arXiv preprint arXiv:1806.00582.

- **Chapter 4: Dynamic & Adaptive Federated Aggregation (Gap 2)**
  (`00-preliminaries/03-dynamic-aggregation-strategies.ipynb`)

  - **Focus**: Empirical evaluation of dynamic aggregation algorithms (FedAvg, FedProx
    with proximal regularization $\\mu$, FedMedian for Byzantine robustness, and
    FedMuon). Assesses accuracy, convergence rates, Pareto efficiency, and inference
    validation on Register No. 2490031.
  - **Literature Citation**:
    - Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
      *Federated Optimization in Heterogeneous Networks (FedProx)*. In Proceedings of
      Machine Learning and Systems (MLSys), 2:429-450.
    - Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). *Byzantine-Robust
      Distributed Learning: Towards Optimal Statistical Rates*. In International
      Conference on Machine Learning (ICML), PMLR 80:5650-5659.
    - Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S.,
      & McMahan, H. B. (2020). *Adaptive Federated Optimization (FedAdam/FedYogi)*. In
      International Conference on Learning Representations (ICLR).

______________________________________________________________________

### Volume II: Cryptographic Privacy Preservation in Distributed Systems

*Target Competency: Implement information-theoretic and computational guarantees to
prevent gradient inversion and reconstruction attacks.*

- **Chapter 5: Differential Privacy Foundations & DP-SGD**
  (`01-privacy-preservation/00-differential-privacy-foundations.ipynb`)

  - **Focus**: Local vs Central Differential Privacy ($\\epsilon, \\delta$).
    Implementing calibrated Gaussian noise addition $\\sigma =
    \\frac{\\sqrt{2\\ln(1.25/\\delta)}}{\\epsilon}$ and gradient clipping threshold $C$
    within decentralized training loops. Privacy tracking via Rényi Differential Privacy
    (RDP).
  - **Literature Citation**:
    - Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). *Calibrating Noise to
      Sensitivity in Private Data Analysis*. In Theory of Cryptography Conference (TCC),
      LNCS 3876:265-284.
    - Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., &
      Zhang, K. (2016). *Deep Learning with Differential Privacy*. In ACM SIGSAC
      Conference on Computer and Communications Security (CCS), 308-318.
    - Mironov, I. (2017). *Rényi Differential Privacy*. In IEEE 30th Computer Security
      Foundations Symposium (CSF), 263-275.

- **Chapter 6: Homomorphic Encryption with Paillier**
  (`01-privacy-preservation/01-homomorphic-encryption-paillier.ipynb`)

  - **Focus**: Additive homomorphic encryption over composite degree residuosity
    classes. Enables parameter aggregation directly on encrypted ciphertexts $c\_{sum} =
    \\prod c_i \\pmod{n^2}$ without revealing client weights to untrusted aggregators.
  - **Literature Citation**:
    - Paillier, P. (1999). *Public-Key Cryptosystems Based on Composite Degree
      Residuosity Classes*. In Advances in Cryptology – EUROCRYPT '99, LNCS
      1592:223-238.
    - Acar, A., Aksu, H., Uluagac, A. S., & Conti, M. (2018). *A Survey on Homomorphic
      Encryption Schemes: Theory and Implementation*. ACM Computing Surveys (CSUR),
      51(4), 1-35.

- **Chapter 7: Secure Multi-Party Computation (SMPC)**
  (`01-privacy-preservation/02-secure-multi-party-computation.ipynb`)

  - **Focus**: Secret sharing schemes including Shamir's $(t, n)$ polynomial threshold
    secret sharing and pairwise additive zero-sum blinding masks. Guarantees zero
    knowledge leakage against honest-but-curious servers with up to $t < n/2$ colluding
    participants.
  - **Literature Citation**:
    - Shamir, A. (1979). *How to Share a Secret*. Communications of the ACM, 22(11),
      612-613.
    - Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S.,
      Ramage, D., Segal, A., & Seth, K. (2017). *Practical Secure Aggregation for
      Privacy-Preserving Machine Learning*. In ACM SIGSAC Conference on Computer and
      Communications Security (CCS), 1175-1191.

- **Chapter 8: Hybrid Defense-in-Depth Benchmark**
  (`01-privacy-preservation/03-hybrid-privacy-preserving-fl-benchmark.ipynb`)

  - **Focus**: Unified framework combining DP, HE, and SMPC. Rigorous Pareto trade-off
    analysis benchmarking communication byte overhead, encryption compute latency, and
    empirical resistance against reconstruction/inversion attacks.
  - **Literature Citation**:
    - Truex, S., Baracaldo, N., Anwar, A., Steinke, T., Ludwig, H., Zhang, R., & Zhou,
      P. (2019). *A Hybrid Approach to Privacy-Preserving Federated Learning*. In
      Proceedings of the 12th ACM Workshop on Artificial Intelligence and Security
      (AISEC), 1-11.
    - Froelicher, D., Troncoso-Pastoriza, J. R., et al. (2021). *Truly
      Privacy-Preserving Federated Analytics for Precision Medicine with Multiparty
      Homomorphic Encryption*. Nature Communications, 12(1), 5910.

______________________________________________________________________

### Volume III: Advanced Geometric Optimization & Matrix Aggregation

*Target Competency: Move beyond Euclidean vector averaging to matrix manifold
optimization and polar decompositions.*

- **Chapter 9: FedMuon: Bias-Corrected Matrix Polar Orthogonalization**
  (`02-federated-aggregation/01-fedmuon-bias-corrected-lmo.ipynb`)

  - **Focus**: Formulation of FedMuon. Replaces standard Euclidean parameter updates
    with polar orthogonalization $W = U H$ computed via quintic Newton-Schulz
    iterations. Corrects client drift and matrix scale distortion under non-identical
    client distributions.
  - **Literature Citation**:
    - Takezawa, Y., Bao, H., Niwa, K., & Sato, I. (2025). *FedMuon: Federated Learning
      with Bias-Corrected Matrix Polar Orthogonalization*. arXiv preprint
      arXiv:2502.14815.
    - Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. Society for
      Industrial and Applied Mathematics (SIAM), Philadelphia.
    - Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).
      *signSGD: Compressed Distributed SGD with Majority Vote*. In International
      Conference on Machine Learning (ICML), PMLR 80:561-570.

- **Chapter 10: Multi-Momentum Aggregation & Baseline Dynamics (FedM3)**
  (`02-federated-aggregation/02-fedm3-vs-baselines.ipynb`)

  - **Focus**: Multi-momentum matrix aggregation (FedM3) vs baseline federated
    optimizers (FedAvg, FedProx, FedAdam). Tracks parameter drift on non-convex
    manifolds and analyzes spectral energy distributions across communication rounds.
  - **Literature Citation**:
    - Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). *Tackling the
      Objective Inconsistency Problem in Heterogeneous Federated Optimization
      (FedNova)*. In Advances in Neural Information Processing Systems (NeurIPS),
      33:7611-7623.
    - Karimireddy et al. (2020), *op. cit.*

______________________________________________________________________

### Volume IV: Continual Learning & Mitigating Catastrophic Forgetting

*Target Competency: Overcome the stability-plasticity dilemma and prevent catastrophic
forgetting across sequentially arriving decentralized tasks.*

- **Chapter 11: Class-Incremental Learning & The Stability-Plasticity Dilemma**
  (`03-continual-learning/01-imagenet-transfer-learning-class-incremental.ipynb`)

  - **Focus**: Class-incremental learning on vision transfer representations. Explores
    backward transfer (BWT), forward transfer (FWT), Fisher information regularization
    (EWC), and the trade-offs of storing exemplar buffers in distributed clients.
  - **Literature Citation**:
    - Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., et al.
      (2017). *Overcoming Catastrophic Forgetting in Neural Networks (EWC)*. Proceedings
      of the National Academy of Sciences (PNAS), 114(13), 3521-3526.
    - Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). *iCaRL:
      Incremental Classifier and Representation Learning*. In IEEE Conference on
      Computer Vision and Pattern Recognition (CVPR), 2001-2010.
    - De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A.,
      Slabaugh, G., & Tuytelaars, T. (2021). *A Continual Learning Survey: Defying
      Forgetting in Classification Tasks*. IEEE Transactions on Pattern Analysis and
      Machine Intelligence (TPAMI), 44(7), 3366-3385.

- **Chapter 12: Vision-TTT: Test-Time Training for Zero Catastrophic Forgetting (Gap
  3)** (`03-continual-learning/02-test-time-training-class-incremental.ipynb`)

  - **Focus**: Evaluates Test-Time Training (Vision-TTT) to eliminate forgetting. Adapts
    network weights online via self-supervised reconstruction objectives at test time
    without requiring clients to retain or communicate private historic replay buffers.
  - **Literature Citation**:
    - Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). *Test-Time
      Training with Self-Supervision for Generalization under Distribution Shifts*. In
      International Conference on Machine Learning (ICML), PMLR 119:9229-9248.
    - Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F.
      (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*.
      arXiv preprint arXiv:2407.04620.

______________________________________________________________________

### Volume V: Frontier Federated Foundation Models & Dynamic Neural Memory

*Target Competency: Synthesize distributed optimization with test-time memorization,
surprise-driven updates, and dual-memory foundation models.*

- **Chapter 13: Self-Modifying Dual-Memory Architectures & Titans at Scale**
  (`04-federated-foundation-models/01-titans-ttt-centralized-vs-federated.ipynb`)
  - **Focus**: Capstone: Implementing and training Titans Dual-Memory Neural Networks in
    centralized vs federated settings. Combines core attention (short-term
    sliding-window memory) with dynamic neural long-term memory weights updated via a
    surprise gradient metric $S_t = \\nabla \\mathcal{L}\_{\\text{memory}}$.
  - **Literature Citation**:
    - Behrouz, A., Pezeshki, C., & Mirrokni, V. (2024). *Titans: Learning to Memorize at
      Test Time*. arXiv preprint arXiv:2412.0009.
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
      Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. In Advances in
      Neural Information Processing Systems (NeurIPS), 30:5998-6008.
    - Ba, J., Hinostroza, R. H., Salakhutdinov, R., & Frey, B. (2016). *Using Fast
      Weights to Attend to the Recent Past*. In Advances in Neural Information
      Processing Systems (NeurIPS), 29:4331-4339.
