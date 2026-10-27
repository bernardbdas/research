"""Generate the 5 Experimental Setup slide cards in PlantUML and compile them."""

import subprocess
from pathlib import Path

out_dir = Path("diagrams/03-system-architecture/slide_presentation")
out_dir.mkdir(parents=True, exist_ok=True)
exp_dir = Path("presentation_assets/experiments")
exp_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# SETUP 1
# -------------------------------------------------------------
setup1_puml = """@startuml setup1_card
top to bottom direction

skinparam monochrome false
skinparam shadowing false
skinparam roundcorner 12
skinparam defaultFontName "Helvetica"
skinparam defaultFontSize 14
skinparam ranksep 35
skinparam nodesep 25

skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #0F172A
    BorderThickness 2.5
    FontStyle bold
    FontSize 16
}

skinparam card {
    BackgroundColor #FFFFFF
    BorderColor #334155
    BorderThickness 1.8
    FontSize 13
}

skinparam arrow {
    Color #0F172A
    Thickness 2.2
    FontSize 13
    FontStyle bold
}

package "EXPERIMENTAL SETUP #1: FEDERATED DATASETS & PARTITIONING TECHNIQUES" as Setup1 #EFF6FF {

    package "1. Multi-Modal Dataset Catalog" as S1 #F8FAFC {
        card "<b>Curated Datasets (Flower Datasets)</b>\\n• <b>MNIST:</b> 70,000 digits (28×28 grayscale)\\n• <b>Fashion-MNIST:</b> 70,000 clothing articles\\n• <b>USPS:</b> 9,298 low-res postal digits\\n• Deterministic Apache Parquet caching" as DataCard #FFFFFF
    }

    package "2. Client Silo Allocation" as S2 #EFF6FF {
        card "<b>Decentralized Edge Network</b>\\n• Total Edge Clients: <b>K = 5 to 10</b>\\n• Local Batch Size: B = 64\\n• Deterministic client seed: 42\\n• No inter-client communication" as ClientCard #FFFFFF
    }

    package "3. Three Evaluated Partition Regimes" as S3 #F0FDF4 {
        card "<b>Regime A: Uniform IID</b>\\n• Every client gets an equal, balanced slice of all classes\\n• Baseline control scenario (No skew)" as RegIID #EFF6FF

        card "<b>Regime B: Dirichlet Label Skew (α = 0.5)</b>\\n• Non-IID Dirichlet distribution: Dir(α)\\n• Realistic heterogeneous label concentrations" as RegDir #FFF7ED

        card "<b>Regime C: Pathological Skew</b>\\n• Each client strictly restricted to <b>2 classes only</b>\\n• Extreme statistical drift stress-test" as RegPath #FEF2F2

        RegIID -down-> RegDir
        RegDir -down-> RegPath
    }

    package "4. Quantitative Target Metrics" as S4 #FAF5FF {
        card "<b>Heterogeneity Measurement</b>\\n• <b>Class Distribution Heatmaps</b>\\n• <b>Shannon Entropy:</b> H(k) = - Σ p<sub>c</sub> log(p<sub>c</sub>)\\n• <b>Total Variation Distance (TVD)</b>" as MetricCard #FFFFFF
    }

    S1 -right-> S2 : Load Partition
    S2 -right-> S3 : Apply Partitioner
    S3 -right-> S4 : Measure Skew
}

@enduml
"""

# -------------------------------------------------------------
# SETUP 2
# -------------------------------------------------------------
setup2_puml = """@startuml setup2_card
top to bottom direction

skinparam monochrome false
skinparam shadowing false
skinparam roundcorner 12
skinparam defaultFontName "Helvetica"
skinparam defaultFontSize 14
skinparam ranksep 35
skinparam nodesep 25

skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #0F172A
    BorderThickness 2.5
    FontStyle bold
    FontSize 16
}

skinparam card {
    BackgroundColor #FFFFFF
    BorderColor #334155
    BorderThickness 1.8
    FontSize 13
}

skinparam arrow {
    Color #0F172A
    Thickness 2.2
    FontSize 13
    FontStyle bold
}

package "EXPERIMENTAL SETUP #2: CENTRALIZED VS. FEDERATED OPTIMIZATION" as Setup2 #EFF6FF {

    package "1. Backbone Architecture" as S1 #F8FAFC {
        card "<b>Pluggable Neural Backbone</b>\\n• Model: <b>SimpleCNN / HOPE-FL Net</b>\\n• Global Average Pooling (GAP)\\n• Footprint: ~126k parameters\\n• Input: (B, 28, 28, 1) → 10 classes" as ModelCard #FFFFFF
    }

    package "2. Paradigm A: Centralized SGD (The Bound)" as S2 #FFF7ED {
        card "<b>Centralized Empirical Ceiling</b>\\n• All training data pooled on 1 central server\\n• Standard SGD with momentum (η = 0.05)\\n• Batch Size: B = 64 across 20 epochs\\n• <b>Zero privacy protection</b>" as CentCard #FFFFFF
    }

    package "3. Paradigm B: Decentralized FedAvg" as S3 #F0FDF4 {
        card "<b>Federated Learning (McMahan 2017)</b>\\n• Data partitioned across <b>K = 5 silos</b>\\n• Local SGD: 1 epoch per round (η = 0.01)\\n• Server: Weighted parameter aggregation\\n• Evaluated across <b>10 communication rounds</b>" as FedCard #FFFFFF
    }

    package "4. Target Comparison Metrics" as S4 #FAF5FF {
        card "<b>The Federation Penalty Measurement</b>\\n• <b>Convergence Lag:</b> Epochs vs. Comm Rounds\\n• <b>Accuracy Deficit:</b> ΔAcc = Acc<sub>Cent</sub> - Acc<sub>Fed</sub>\\n• <b>Overlay Loss Trajectory Analysis</b>" as MetricCard #FFFFFF
    }

    S1 -right-> S2 : Benchmark 1
    S1 -right-> S3 : Benchmark 2
    S2 -down-> S4 : Central Ceiling
    S3 -down-> S4 : FedAvg Trajectory
}

@enduml
"""

# -------------------------------------------------------------
# SETUP 3
# -------------------------------------------------------------
setup3_puml = """@startuml setup3_card
top to bottom direction

skinparam monochrome false
skinparam shadowing false
skinparam roundcorner 12
skinparam defaultFontName "Helvetica"
skinparam defaultFontSize 14
skinparam ranksep 35
skinparam nodesep 25

skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #0F172A
    BorderThickness 2.5
    FontStyle bold
    FontSize 16
}

skinparam card {
    BackgroundColor #FFFFFF
    BorderColor #334155
    BorderThickness 1.8
    FontSize 13
}

skinparam arrow {
    Color #0F172A
    Thickness 2.2
    FontSize 13
    FontStyle bold
}

package "EXPERIMENTAL SETUP #3: IID VS. NON-IID DATA & THE CLIENT DRIFT PHENOMENON" as Setup3 #EFF6FF {

    package "1. Dataset Testbed" as S1 #F8FAFC {
        card "<b>Multi-Domain Testbed</b>\\n• <b>MNIST:</b> Clean handwritten digits\\n• <b>Fashion-MNIST:</b> Complex visual textures\\n• <b>USPS:</b> Low-resolution edge images" as DataCard #FFFFFF
    }

    package "2. Three Evaluated Distribution Regimes" as S2 #FFF7ED {
        card "<b>Regime 1: Uniform IID</b>\\nEqual class balance (No drift driver)" as R1 #FFFFFF

        card "<b>Regime 2: Dirichlet Skew (α = 0.5)</b>\\nModerate Non-IID label concentration" as R2 #EFF6FF

        card "<b>Regime 3: Pathological (2-Class)</b>\\nExtreme Non-IID (Clients disagree completely)" as R3 #FEF2F2

        R1 -down-> R2
        R2 -down-> R3
    }

    package "3. Validation Engine" as S3 #F0FDF4 {
        card "<b>Multi-Round FL Simulation</b>\\n• Backbone: <b>SimpleCNN / HOPE-FL</b>\\n• Clients: K = 5 Silos, Rounds = 10\\n• Strategy: Standard FedAvg\\n• Evaluates divergence of local parameter updates" as SimCard #FFFFFF
    }

    package "4. Measured Target Outputs" as S4 #FAF5FF {
        card "<b>Client Drift Impact</b>\\n• <b>Convergence Trajectory Drop (%)</b>\\n• <b>Local Gradient Divergence Angle</b>\\n• <b>Real-Time Sequence Test:</b> [2, 4, 9, 0, 0, 3, 1]" as OutCard #FFFFFF
    }

    S1 -right-> S2 : Partition
    S2 -right-> S3 : Run FL
    S3 -right-> S4 : Quantify Drift Drop
}

@enduml
"""

# -------------------------------------------------------------
# SETUP 4
# -------------------------------------------------------------
setup4_puml = """@startuml setup4_card
top to bottom direction

skinparam monochrome false
skinparam shadowing false
skinparam roundcorner 12
skinparam defaultFontName "Helvetica"
skinparam defaultFontSize 14
skinparam ranksep 35
skinparam nodesep 25

skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #0F172A
    BorderThickness 2.5
    FontStyle bold
    FontSize 16
}

skinparam card {
    BackgroundColor #FFFFFF
    BorderColor #334155
    BorderThickness 1.8
    FontSize 13
}

skinparam arrow {
    Color #0F172A
    Thickness 2.2
    FontSize 13
    FontStyle bold
}

package "EXPERIMENTAL SETUP #4: STATIC VS. DYNAMIC AGGREGATION STRATEGIES" as Setup4 #EFF6FF {

    package "1. Non-IID Client Cohort" as S1 #F8FAFC {
        card "<b>Heterogeneous Edge Testbed</b>\\n• Dataset: MNIST under Dirichlet (α = 0.5)\\n• K = 5 Edge Clients, Rounds = 10\\n• Standardized client learning rate: η = 0.01" as SiloCard #FFFFFF
    }

    package "2. Evaluated Server Aggregation Algorithms" as S2 #F0FDF4 {
        card "<b>Static Baseline: FedAvg (McMahan 2017)</b>\\nArithmetic sample-weighted mean (No drift correction)" as A_FedAvg #FFFFFF

        card "<b>Proximal Dynamic: FedProx (Li 2020)</b>\\nProximal loss penalty (μ = 0.01) to constrain local drift" as A_FedProx #EFF6FF

        card "<b>Robust Dynamic: FedMedian / FedTrimmedMean</b>\\nCoordinate-wise median & trimmed mean (β = 0.1) for Byzantine filtering" as A_FedMed #FFF7ED

        card "<b>Adaptive Dynamic: FedAdam (Reddi 2020)</b>\\nServer coordinate momentum (η = 0.05, β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.99)" as A_FedAdam #FAF5FF

        A_FedAvg -down-> A_FedProx
        A_FedProx -down-> A_FedMed
        A_FedMed -down-> A_FedAdam
    }

    package "3. Target Pareto Dimensions" as S3 #FAF5FF {
        card "<b>Multi-Objective Evaluation Matrix</b>\\n• <b>Non-IID Accuracy Trajectory (%)</b>\\n• <b>Computational Complexity (FLOPs / Round)</b>\\n• <b>Cryptographic Overhead (Paillier HE & SecAgg)</b>\\n• <b>Pareto Frontier Efficiency Synthesis</b>" as MatrixCard #FFFFFF
    }

    S1 -right-> S2 : Client Deltas
    S2 -right-> S3 : Multi-Metric Benchmark
}

@enduml
"""

# -------------------------------------------------------------
# SETUP 5
# -------------------------------------------------------------
setup5_puml = """@startuml setup5_card
top to bottom direction

skinparam monochrome false
skinparam shadowing false
skinparam roundcorner 12
skinparam defaultFontName "Helvetica"
skinparam defaultFontSize 14
skinparam ranksep 35
skinparam nodesep 25

skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #0F172A
    BorderThickness 2.5
    FontStyle bold
    FontSize 16
}

skinparam card {
    BackgroundColor #FFFFFF
    BorderColor #334155
    BorderThickness 1.8
    FontSize 13
}

skinparam arrow {
    Color #0F172A
    Thickness 2.2
    FontSize 13
    FontStyle bold
}

package "EXPERIMENTAL SETUP #5: FEDM3 VS. FEDMUON & BASELINE OPTIMIZERS" as Setup5 #EFF6FF {

    package "1. Edge Architecture Testbed" as S1 #F8FAFC {
        card "<b>Neural Models Compared</b>\\n• <b>Standard Baseline:</b> SimpleCNN (Static trunk)\\n• <b>Our Architecture:</b> HOPE-FL + CMS\\n  <i>(Continuous test-time associative memory)</i>" as ArchCard #FFFFFF
    }

    package "2. Advanced Manifold Optimizers Compared" as S2 #FAF5FF {
        card "<b>FedMuon Baseline (Takezawa et al. 2025)</b>\\n• Single-scale temporal momentum (β = 0.9)\\n• Newton-Schulz polar orthogonalization\\n• Unconditioned on client drift" as MuonCard #FFFFFF

        card "<b>FedM3 Optimizer (Ours: Module #1)</b>\\n• <b>Dual-Scale Momentum:</b> Fast (β<sub>f</sub>=0.80) + Slow (β<sub>s</sub>=0.98)\\n• <b>Convex Blending:</b> M<sub>t</sub> = 0.6 V<sub>fast</sub> + 0.4 V<sub>slow</sub>\\n• <b>Quintic Newton-Schulz:</b> Polar factor on St(p, n)" as M3Card #ECFDF5

        card "<b>Classical Baselines:</b> FedAvg, FedProx, FedAdam" as ClassicCard #EFF6FF

        ClassicCard -down-> MuonCard
        MuonCard -down-> M3Card
    }

    package "3. Target Validation Matrix" as S3 #F0FDF4 {
        card "<b>Empirical Benchmarks (Slide 39)</b>\\n• <b>Non-IID Accuracy Trajectory across Rounds</b>\\n• <b>Loss Surface Optimization Stability</b>\\n• <b>Singular Value Spectrum Normalization</b>\\n• <b>Rounds to Cross 90% Target Accuracy</b>" as BenchCard #FFFFFF
    }

    S1 -right-> S2 : Evaluated on Silos (α=0.5)
    S2 -right-> S3 : Multi-Optimizer Benchmark
}

@enduml
"""

cards = {
    "slide_31_setup1_partitioning": setup1_puml,
    "slide_33_setup2_centralized_vs_fed": setup2_puml,
    "slide_35_setup3_iid_vs_noniid": setup3_puml,
    "slide_37_setup4_dynamic_aggregation": setup4_puml,
    "slide_39_setup5_fedm3_vs_fedmuon": setup5_puml,
}

for name, content in cards.items():
    # Replace @startuml <name> with @startuml <actual_filename>
    lines = content.split("\n")
    lines[0] = f"@startuml {name}"
    fixed_content = "\n".join(lines)
    
    puml_path = out_dir / f"{name}.puml"
    with open(puml_path, "w") as f:
        f.write(fixed_content)
    # compile with plantuml
    subprocess.run(["plantuml", "-tsvg", str(puml_path)], check=True)
    svg_path = out_dir / f"{name}.svg"
    png_path = exp_dir / f"{name}.png"
    subprocess.run(["rsvg-convert", "-w", "1600", str(svg_path), "-o", str(png_path)], check=True)
    print(f"Generated setup card: {png_path}")
