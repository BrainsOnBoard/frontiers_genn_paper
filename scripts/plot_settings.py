import seaborn as sns
import sys

presentation = "presentation" in sys.argv[1:]

# Set the plotting style
if presentation:
    sns.set(context="talk")
    sns.set_style("whitegrid", {"font.family":"sans-serif", "font.sans-serif":"Verdana"})
else:
    sns.set(context="paper")
    sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times New Roman"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 0.1})

mm_to_inches = 0.039370079
column_width = 85.0 * mm_to_inches
double_column_width = 180.0 * mm_to_inches