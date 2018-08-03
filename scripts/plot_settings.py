import seaborn as sns

# Set the plotting style
sns.set(context="paper")
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times New Roman"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 0.1})

mm_to_inches = 0.039370079
column_width = 85.0 * mm_to_inches
double_column_width = 180.0 * mm_to_inches