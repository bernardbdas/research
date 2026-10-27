# Decentralized Intelligence: From Statistical Foundations to Test-Time Continual Learning

> **Sequential Research Curriculum for Federated Machine Learning**
>
> *Thirteen target-achieving chapters organized across five thematic categories: from
> foundational distributed data mechanics to test-time neural memory foundation models.
> Executed in a publication-grade, high-contrast visual and mathematical style.*

______________________________________________________________________

## 📁 Curriculum Architecture: 5 Thematic Pillars (Chapters 01–13)

```text
notebooks/
├── 01-federated-foundations/             # Pillar 1: Distributed Data Mechanics & Baselines
│   ├── 01-data-partitioning.ipynb                    [Chapter 01: Distributional Skew & Multi-Modal Catalog]
│   ├── 02-federation-penalty.ipynb                   [Chapter 02: Centralized Bound vs FedAvg Penalty]
│   ├── 03-client-drift.ipynb                         [Chapter 03: Statistical Heterogeneity & Real-Time Drift]
│   └── 04-dynamic-aggregation.ipynb                  [Chapter 04: Dynamic & Adaptive Server Aggregation]
├── 02-privacy-preservation/              # Pillar 2: Cryptographic Protocols & Differential Privacy
│   ├── 05-differential-privacy.ipynb                 [Chapter 05: Calibrated DP-SGD & RDP Accounting]
│   ├── 06-homomorphic-encryption.ipynb               [Chapter 06: Paillier Additive Homomorphic Encryption]
│   ├── 07-secure-multi-party-computation.ipynb       [Chapter 07: Shamir Secret Sharing & SecAgg Masking]
│   └── 08-hybrid-privacy-benchmark.ipynb             [Chapter 08: Defense-in-Depth Privacy Trilemma Benchmark]
├── 03-federated-aggregation/             # Pillar 3: Matrix Optimization & Manifold Aggregation
│   ├── 09-fedmuon-matrix-optimization.ipynb          [Chapter 09: Newton-Schulz Polar Orthogonalization]
│   └── 10-fedm3-multimomentum-benchmark.ipynb        [Chapter 10: Multi-Momentum Aggregation & HOPE-FL]
├── 04-continual-learning/                # Pillar 4: Non-Stationary Drift & Test-Time Adaptation
│   ├── 11-class-incremental-continual-learning.ipynb [Chapter 11: Vision Transfer & Stability-Plasticity]
│   └── 12-test-time-training-continual-learning.ipynb[Chapter 12: Vision-TTT Self-Supervised Adaptation]
└── 05-federated-foundation-models/       # Pillar 5: Neural Memory Architectures at Scale
    ├── 13-titans-neural-memory-benchmark.ipynb       [Chapter 13: Dual-Memory Titans & Surprise Gradients]
    └── 14-nested-hope-fl-fedm3-benchmark.ipynb       [Chapter 14: Truly Nested FL: HOPE-FL, CMS & FedM3]
```

______________________________________________________________________

## Publication Graphic & Code Standards

Each notebook in this curriculum adheres to strict, publication-grade standards:

1. **Vocal & Target-Achieving Structure**:
   - `## **1. [TARGET] Problem Statement & Quantitative Bound**`
   - `## **2. [MECHANICS] Mathematical Architecture & Data Flow**` (with Mermaid
     flowchart)
   - `## **3. [BENCHMARK] Execution & Measurement**`
   - `## **4. [VERDICT] Empirical Performance Matrix & Failure Modes**`
1. **High-Contrast Publication Visualizations**:
   - Vivid, high-contrast, modern palettes (`#2b5c8f`, `#d95f02`, `#7570b3`, `#e7298a`,
     `#1b9e77`, `#2ca02c`, `#3498db`, `#e67e22`).
   - Clean white / light backgrounds under dark bold typography for sharp contrast.
   - Sharp geometric gridlines (`linestyle="--"`, `linewidth=0.8`, `alpha=0.35`,
     `color="#555555"`).
   - Generous plot sizing (`figsize=(14, 6)` or `(16, 6)`) with explicit labels and
     padded titles.
1. **Embedded Mermaid Flowcharts**:
   - High-clarity architecture and protocol diagrams embedded directly in Markdown
     cells.
1. **Architectural Cleanliness**:
   - Zero custom class definitions inside notebook cells.
   - All models, aggregators, and data pipelines live in `models/`, `federated/`, and
     `utils/`.
   - Pluggable model backbones via `BACKBONE_NAME` and `models.get_model()`.
1. **Mandatory Formal Scholarly Citations**:
   - Every theoretical claim, complexity bound, and optimization algorithm is explicitly
     grounded in peer-reviewed literature.

______________________________________________________________________

## Detailed Chapter Syllabus & Literature Grounding

### **Pillar 1: Federated Foundations & Mechanics (`01-federated-foundations/`)**

#### **Chapter 01: Decentralized Ingestion & Non-IID Partitioning**

- **File**:
  \[`01-data-partitioning.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/01-federated-foundations/01-data-partitioning.ipynb)
- **Target**: Benchmark and visualize statistical data partitioning across client
  federations under 3 distinct regimes: Uniform IID, Dirichlet Non-IID label skew
  ($\\text{Dir}(\\alpha)$), and Pathological 2-class allocation across a multi-modal
  catalog (15 Vision, 4 Audio, 5 Tabular, 9 Text, 1 Multimodal).
- **Citations**:
  - McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
    *Communication-Efficient Learning of Deep Networks from Decentralized Data*.
    AISTATS.
  - Hsu, T. M. H., Qi, H., & Brown, M. (2019). *Measuring the Effects of Non-Identical
    Distributions on Federated Visual Classification*. arXiv:1909.06335.

#### **Chapter 02: Centralized vs. Federated Optimization & The Federation Penalty**

- **File**:
  \[`02-federation-penalty.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/01-federated-foundations/02-federation-penalty.ipynb)
- **Target**: Quantify the "Federation Penalty" (the communication cost, convergence
  lag, and loss discrepancy between centralized SGD and decentralized FedAvg).
- **Citations**:
  - McMahan et al. (2017), *op. cit.*
  - Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). *Federated Learning:
    Challenges, Methods, and Future Directions*. IEEE Signal Processing Magazine.

#### **Chapter 03: Statistical Heterogeneity & The Client Drift Phenomenon**

- **File**:
  \[`03-client-drift.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/01-federated-foundations/03-client-drift.ipynb)
- **Target**: Quantify client parameter drift under Dirichlet non-identical
  distributions ($\\alpha=0.5$) and pathological partition allocations across MNIST,
  USPS, and Fashion-MNIST with real-time target sequence tracking.
- **Citations**:
  - Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T.
    (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML.
  - Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). *Federated
    Learning with Non-IID Data*. arXiv:1806.00582.

#### **Chapter 04: Dynamic & Adaptive Federated Aggregation**

- **File**:
  \[`04-dynamic-aggregation.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/01-federated-foundations/04-dynamic-aggregation.ipynb)
- **Target**: Benchmark dynamic and adaptive aggregation strategies (FedAvg, FedProx,
  FedMedian, FedTrimmedMean, FedAdam, FedMuon) on Pareto efficiency frontiers.
- **Citations**:
  - Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
    *Federated Optimization in Heterogeneous Networks (FedProx)*. MLSys.
  - Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). *Byzantine-Robust Distributed
    Learning: Towards Optimal Statistical Rates*. ICML.
  - Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., &
    McMahan, H. B. (2020). *Adaptive Federated Optimization*. ICLR.

______________________________________________________________________

### **Pillar 2: Privacy Preservation & Cryptographic Protocols (`02-privacy-preservation/`)**

#### **Chapter 05: Differential Privacy Foundations & DP-SGD**

- **File**:
  \[`05-differential-privacy.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/02-privacy-preservation/05-differential-privacy.ipynb)
- **Target**: Formulate local vs central $(\\epsilon, \\delta)$-Differential Privacy,
  gradient clipping threshold $C$, Gaussian perturbation noise $\\sigma$, and Rényi DP
  tracking.
- **Citations**:
  - Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). *Calibrating Noise to
    Sensitivity in Private Data Analysis*. TCC.
  - Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., &
    Zhang, K. (2016). *Deep Learning with Differential Privacy*. ACM CCS.
  - Mironov, I. (2017). *Rényi Differential Privacy*. IEEE CSF.

#### **Chapter 06: Homomorphic Encryption with Paillier**

- **File**:
  \[`06-homomorphic-encryption.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/02-privacy-preservation/06-homomorphic-encryption.ipynb)
- **Target**: Implement additive homomorphic parameter aggregation over composite degree
  residuosity classes with fixed-point encoding and zero server parameter visibility.
- **Citations**:
  - Paillier, P. (1999). *Public-Key Cryptosystems Based on Composite Degree Residuosity
    Classes*. EUROCRYPT.
  - Acar, A., Aksu, H., Uluagac, A. S., & Conti, M. (2018). *A Survey on Homomorphic
    Encryption Schemes: Theory and Implementation*. ACM CSUR.

#### **Chapter 07: Secure Multi-Party Computation (SMPC)**

- **File**:
  \[`07-secure-multi-party-computation.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/02-privacy-preservation/07-secure-multi-party-computation.ipynb)
- **Target**: Implement Shamir $(t, n)$ polynomial secret sharing and Bonawitz pairwise
  zero-sum masking for exact reconstruction under honest-but-curious adversaries.
- **Citations**:
  - Shamir, A. (1979). *How to Share a Secret*. CACM.
  - Bonawitz, K., et al. (2017). *Practical Secure Aggregation for Privacy-Preserving
    Machine Learning*. ACM CCS.

#### **Chapter 08: Hybrid Defense-in-Depth Benchmark**

- **File**:
  \[`08-hybrid-privacy-benchmark.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/02-privacy-preservation/08-hybrid-privacy-benchmark.ipynb)
- **Target**: Multi-objective Pareto evaluation across 4 privacy regimes (Unprotected
  Baseline, Pure DP-SGD, Pure SecAgg, Hybrid DP+SecAgg).
- **Citations**:
  - Truex, S., Baracaldo, N., et al. (2019). *A Hybrid Approach to Privacy-Preserving
    Federated Learning*. ACM AISEC.
  - Froelicher, D., et al. (2021). *Truly Privacy-Preserving Federated Analytics with
    Multiparty Homomorphic Encryption*. Nature Communications.

______________________________________________________________________

### **Pillar 3: Advanced Matrix Optimization & Aggregation (`03-federated-aggregation/`)**

#### **Chapter 09: FedMuon: Bias-Corrected Matrix Polar Orthogonalization**

- **File**:
  \[`09-fedmuon-matrix-optimization.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/03-federated-aggregation/09-fedmuon-matrix-optimization.ipynb)
- **Target**: Quintic Newton-Schulz matrix polar orthogonalization $W = U H$ replacing
  Euclidean averaging on non-convex manifolds under high heterogeneity.
- **Citations**:
  - Takezawa, Y., Bao, H., Niwa, K., & Sato, I. (2025). *FedMuon: Federated Learning
    with Bias-Corrected Matrix Polar Orthogonalization*. arXiv:2502.14815.
  - Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.

#### **Chapter 10: Multi-Momentum Aggregation & Baseline Dynamics (FedM3)**

- **File**:
  \[`10-fedm3-multimomentum-benchmark.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/03-federated-aggregation/10-fedm3-multimomentum-benchmark.ipynb)
- **Target**: Multi-Momentum Matrix Aggregation (FedM3) vs FedAvg, FedProx, FedAdam, and
  FedMuon on the HOPE-FL architecture with Continuous Memory System (CMS).
- **Citations**:
  - Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). *Tackling the
    Objective Inconsistency Problem in Heterogeneous Federated Optimization (FedNova)*.
    NeurIPS.
  - Karimireddy et al. (2020), *op. cit.*

______________________________________________________________________

### **Pillar 4: Continual & Test-Time Adaptation (`04-continual-learning/`)**

#### **Chapter 11: Class-Incremental Learning & The Stability-Plasticity Dilemma**

- **File**:
  \[`11-class-incremental-continual-learning.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/04-continual-learning/11-class-incremental-continual-learning.ipynb)
- **Target**: Evaluate catastrophic forgetting, Backward Transfer (BWT), Forward
  Transfer (FWT), and memory replay on sequential vision transfer tasks.
- **Citations**:
  - Kirkpatrick, J., et al. (2017). *Overcoming Catastrophic Forgetting in Neural
    Networks (EWC)*. PNAS.
  - Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). *iCaRL:
    Incremental Classifier and Representation Learning*. CVPR.

#### **Chapter 12: Vision-TTT: Test-Time Training for Zero Catastrophic Forgetting**

- **File**:
  \[`12-test-time-training-continual-learning.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/04-continual-learning/12-test-time-training-continual-learning.ipynb)
- **Target**: Test-Time Training (Vision-TTT) with self-supervised reconstruction for
  test-time adaptation without retaining historical replay exemplars.
- **Citations**:
  - Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). *Test-Time
    Training with Self-Supervision for Generalization under Distribution Shifts*. ICML.
  - Sun, Y., et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive
    Hidden States*. arXiv:2407.04620.

______________________________________________________________________

### **Pillar 5: Federated Foundation Models & Neural Memory (`05-federated-foundation-models/`)**

#### **Chapter 13: Self-Modifying Dual-Memory Architectures & Titans at Scale**

- **File**:
  \[`13-titans-neural-memory-benchmark.ipynb`\](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/05-federated-foundation-models/13-titans-neural-memory-benchmark.ipynb)
- **Target**: Dual-memory neural foundation models (Titans with core short-term
  attention + neural long-term fast weights updated via surprise metric $S_t = \\nabla
  \\mathcal{L}\_{\\text{memory}}$) in centralized vs federated regimes.
- **Citations**:
  - Behrouz, A., Pezeshki, C., & Mirrokni, V. (2024). *Titans: Learning to Memorize at
    Test Time*. arXiv:2412.0009.
  - Ba, J., Hinostroza, R. H., Salakhutdinov, R., & Frey, B. (2016). *Using Fast Weights
    to Attend to the Recent Past*. NeurIPS.

#### **Chapter 14: Truly Nested Federated Learning (HOPE-FL, CMS & FedM3)**

- **File**:
  [`14-nested-hope-fl-fedm3-benchmark.ipynb`](file:///Users/bernard/Workspace/ACADEMIA/research/notebooks/05-federated-foundation-models/14-nested-hope-fl-fedm3-benchmark.ipynb)
- **Target**: Comprehensive empirical benchmark evaluating Truly Nested Federated Learning:
  Module #1 (FedM3 server dynamic aggregation on the Stiefel manifold), Module #2 (Continuous
  Memory System forward-pass test-time adaptation under covariate shifts), and Module #3
  (HOPE-FL end-to-end synergy) compared against FedAvg, FedProx, FedAdam, and FedMuon.
- **Citations**:
  - Takezawa, Y., et al. (2025). *FedMuon: Federated Learning with Bias-Corrected Matrix Polar Orthogonalization*. arXiv:2502.14815.
  - Behrouz, A., et al. (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2412.0009.
  - Sun, Y., et al. (2020). *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*. ICML.
  - Karimireddy, S. P., et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML.
