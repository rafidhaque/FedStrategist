graph TD
    subgraph "Federated Learning Environment"
        C1(Client 1)
        C2(Client 2)
        C3(Client ...)
        CN(Client N)
    end

    subgraph "Central Server (FedStrategist Framework)"
        A["1&#46; Collect Updates <br/> [Δw₁, Δw₂, ...]"]
        B["2&#46; Instrumentation Layer <br/> Compute State Vector S_t"]
        C["3&#46; Meta-Learning Agent <br/> (Contextual Bandit)"]
        D{"4&#46; Choose Rule 'j' <br/> from Arsenal"}
        E["5&#46; Apply Rule 'j' <br/> to Aggregate Updates"]
        F[Global Model W_t]
        G["6&#46; Calculate Reward R_t <br/> R = ΔAcc - λC_j"]
    end

    subgraph "Defense Arsenal"
        R1[FedAvg]
        R2[Median]
        R3[Krum]
    end

    %% Data Flow
    C1 -- "Δw₁" --> A
    C2 -- "Δw₂" --> A
    CN -- "Δwₙ" --> A

    A --> B
    B -- "State S_t" --> C
    C --> D
    
    D -- "Selects FedAvg" --> R1
    D -- "Selects Median" --> R2
    D -- "Selects Krum" --> R3
    
    R1 -- "Chosen Rule" --> E
    R2 -- "Chosen Rule" --> E
    R3 -- "Chosen Rule" --> E
    A -- "Updates [Δw]" --> E

    E --> F
    
    F -- "New Accuracy" --> G
    B -- "Old Accuracy (from prev. round)" --> G
    
    G -- "Update Agent with (S_t, a_t, R_t)" --> C

    %% Styling
    style Server fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style R1 fill:#cfc,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style R2 fill:#cfc,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style R3 fill:#cfc,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5