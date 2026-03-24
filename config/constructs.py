"""
Construct Mapping Configuration

Defines the mapping between survey items and the conceptual model constructs:

Independent Variables (IV):
    - Firm Active Engagement Constructs

Mediator Variables (M):
    - Structural Capital
    - Relational Capital
    - Cognitive Capital
    - Trust

Dependent Variables (DV):
    - Community Engagement Constructs

Moderator Variables:
    - Social versus Functional Needs
    - Need for Self-Esteem Enhancement

Reference: Conceptual Model from the research design.
"""

# ---------------------------------------------------------------------------
# Independent Variables – Firm Active Engagement
# ---------------------------------------------------------------------------
INDEPENDENT_VARIABLES = {
    "firm_active_engagement": {
        "label": "Firm Active Engagement",
        "items": [
            "FAE_1", "FAE_2", "FAE_3", "FAE_4", "FAE_5",
        ],
        "description": (
            "Measures the extent to which the firm actively engages "
            "with its online brand community members."
        ),
    },
}

# ---------------------------------------------------------------------------
# Mediator Variables – Social Capital Dimensions + Trust
# ---------------------------------------------------------------------------
MEDIATOR_VARIABLES = {
    "structural_capital": {
        "label": "Structural Capital",
        "items": [
            "SC_1", "SC_2", "SC_3", "SC_4", "SC_5",
        ],
        "description": (
            "Network ties and structural connections among community members."
        ),
    },
    "relational_capital": {
        "label": "Relational Capital",
        "items": [
            "RC_1", "RC_2", "RC_3", "RC_4", "RC_5",
        ],
        "description": (
            "Quality of relationships, norms, and identification "
            "within the community."
        ),
    },
    "cognitive_capital": {
        "label": "Cognitive Capital",
        "items": [
            "CC_1", "CC_2", "CC_3", "CC_4", "CC_5",
        ],
        "description": (
            "Shared language, codes, and narratives among community members."
        ),
    },
    "trust": {
        "label": "Trust",
        "items": [
            "TR_1", "TR_2", "TR_3", "TR_4", "TR_5",
        ],
        "description": (
            "Trust in the brand and in fellow community members."
        ),
    },
}

# ---------------------------------------------------------------------------
# Dependent Variables – Community Engagement
# ---------------------------------------------------------------------------
DEPENDENT_VARIABLES = {
    "community_engagement": {
        "label": "Community Engagement",
        "items": [
            "CE_1", "CE_2", "CE_3", "CE_4", "CE_5",
        ],
        "description": (
            "Degree of member engagement behaviours within the "
            "online brand community."
        ),
    },
}

# ---------------------------------------------------------------------------
# Moderator Variables
# ---------------------------------------------------------------------------
MODERATOR_VARIABLES = {
    "social_functional_needs": {
        "label": "Social versus Functional Needs",
        "items": [
            "SFN_1", "SFN_2", "SFN_3", "SFN_4", "SFN_5",
        ],
        "description": (
            "Whether community participation is driven by social "
            "needs or functional/utilitarian needs."
        ),
    },
    "self_esteem_enhancement": {
        "label": "Need for Self-Esteem Enhancement",
        "items": [
            "SE_1", "SE_2", "SE_3", "SE_4", "SE_5",
        ],
        "description": (
            "The degree to which members seek self-esteem enhancement "
            "through community participation."
        ),
    },
}

# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
ALL_CONSTRUCTS = {
    **INDEPENDENT_VARIABLES,
    **MEDIATOR_VARIABLES,
    **DEPENDENT_VARIABLES,
    **MODERATOR_VARIABLES,
}


def get_all_items():
    """Return a flat list of every survey item across all constructs."""
    items = []
    for construct in ALL_CONSTRUCTS.values():
        items.extend(construct["items"])
    return items


def get_construct_items(construct_key):
    """Return the list of survey-item column names for *construct_key*."""
    return ALL_CONSTRUCTS[construct_key]["items"]


def get_construct_label(construct_key):
    """Return the human-readable label for *construct_key*."""
    return ALL_CONSTRUCTS[construct_key]["label"]
