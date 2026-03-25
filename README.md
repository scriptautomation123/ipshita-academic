# Social Capital & Community Engagement Analysis

Statistical analysis for a mediation–moderation model examining how **Firm
Active Engagement** influences **Community Engagement** through **Social Capital
dimensions** (Structural Capital, Relational Capital, Cognitive Capital, and
Trust), moderated by **Social vs Functional Needs** and **Need for Self-Esteem
Enhancement**.

## Conceptual Model

```
                        ┌──────────────────┐
                        │ Structural Capital│
                        ├──────────────────┤
  ┌────────────────┐    │ Relational Capital│    ┌────────────────────┐
  │  Firm Active   │───▶├──────────────────┤───▶│    Community       │
  │  Engagement    │    │ Cognitive Capital │    │    Engagement      │
  │  (IV)          │───▶├──────────────────┤───▶│    (DV)            │
  └────────────────┘    │     Trust         │    └────────────────────┘
         │              └──────────────────┘              │
         │                 (Mediators)                    │
         │                                                │
         └──────────────── Moderated by ──────────────────┘
                    • Social vs Functional Needs
                    • Need for Self-Esteem Enhancement
```

### Variable Roles

| Role               | Constructs                                                         |
|--------------------|--------------------------------------------------------------------|
| **Independent (IV)** | Firm Active Engagement                                           |
| **Mediators (M)**    | Structural Capital, Relational Capital, Cognitive Capital, Trust |
| **Dependent (DV)**   | Community Engagement                                             |
| **Moderators**       | Social vs Functional Needs, Need for Self-Esteem Enhancement     |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### With real survey data

Place the Excel data file (`.xlsx`) in the `data/` directory, then run:

```bash
python run_analysis.py
```

Or specify a file path directly:

```bash
python run_analysis.py --file path/to/data.xlsx
```

### With synthetic sample data (demonstration)

```bash
python run_analysis.py --sample
python run_analysis.py --sample --sample-n 500  # custom sample size
```

## Analyses Performed

1. **Descriptive Statistics** – mean, SD, skewness, kurtosis for all constructs
2. **Reliability Analysis** – Cronbach's alpha for each multi-item scale
3. **Correlation Matrix** – Pearson correlations with significance values and heat-map
4. **Mediation Analysis** – Baron & Kenny method with Sobel test for each mediator path
5. **Moderation Analysis** – interaction effects of each moderator on the IV → DV link
6. **Moderated Mediation** – conditional indirect effects at low / mean / high moderator levels

## Output

All results are saved to the `output/` directory:

| File                               | Contents                                |
|------------------------------------|-----------------------------------------|
| `descriptive_statistics.csv`       | Descriptive stats for all constructs    |
| `reliability.csv`                  | Cronbach's alpha per construct          |
| `correlation_matrix.csv`           | Pearson correlation matrix              |
| `correlation_matrix.png`           | Correlation heat-map                    |
| `mediation_results.csv`            | Baron & Kenny paths + Sobel test        |
| `moderation_results.csv`           | Interaction coefficients                |
| `moderated_mediation_results.csv`  | Conditional indirect effects            |

## Configuration

Construct-to-item mappings are defined in `config/constructs.py`. Update the
item lists to match the actual column names in your survey data file.

## Project Structure

```
├── README.md
├── requirements.txt
├── run_analysis.py              # Entry point
├── config/
│   └── constructs.py            # Construct ↔ survey-item mappings
├── analysis/
│   ├── data_loader.py           # Data loading & composite scores
│   └── mediation_moderation_analysis.py  # Full analysis pipeline
└── output/                      # Generated results (git-ignored)
```