from dataclasses import asdict

from lida.datamodel import Goal

class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(
        self,
    ) -> None:

        pass

    def get_template(self, goal: Goal, library: str):

        general_instructions = f"""If the solution requires a single value (e.g. max, min, median, first, last etc), ALWAYS add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F). If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column i) convert date fields to date types using data[''] = pd.to_datetime(data[<field>], errors='coerce'), ALWAYS use  errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[<field>])] iii) convert field to right time format for plotting.  ALWAYS make sure the x-axis labels are legible and visible (e.g., rotate the text and make it smaller when needed, make sure that the text is always within bounds). Solve the task  carefully by completing ONLY the <imports> AND <stub> section. Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}. DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable data. ALWAYS make sure that `def plot(data: pd.DataFrame):` is included and that only <imports> and <stub> are modified.
        
        If the fields are groupable (check their properties in the summary (they are groupable if the property "groupable" == True).
        Group by and aggregate them properly and prevent 'cannot reindex on an axis with duplicate labels' errors (e.g. trying to plot values on the dates where there are multuple values with the same date will bedata=data.groupby('DATE').agg("""+"""{"""+"""'SOME_COLUMN':'mean'"""+"""}""" +""")).
        """

        matplotlib_instructions = f""" {general_instructions} DO NOT include plt.show(). The plot method must return a matplotlib object (plt). Think step by step. \n"""

        if library == "matplotlib":
            instructions = {
                "role": "assistant",
                "content": f"  {matplotlib_instructions}. Use BaseMap for charts that require a map. "}
            template = \
                f"""

# ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED. ALWAYS INCLUDE THE FUNCTION DEFINITION. 
import matplotlib.pyplot as plt
import pandas as pd
#testing comment
<imports> # only modify this section
# plan - 
def plot(data: pd.DataFrame): 
    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt;

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""
        elif library == "seaborn":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions}. Use BaseMap for charts that require a map. "}

            template = \
                f"""
# ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED. ALWAYS INCLUDE THE FUNCTION DEFINITION. 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
<imports> # only modify this section
# solution plan
# i.  ..
def plot(data: pd.DataFrame): #ALWAYS INCLUDE THIS.

    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt;

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""

        elif library == "ggplot":
            instructions = {
                "role": "assistant",
                "content": f"{general_instructions}. The plot method must return a ggplot object (chart)`. Think step by step.p. \n",
            }

            template = \
                f"""
# ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED. ALWAYS INCLUDE THE FUNCTION DEFINITION. 
import plotnine as p9
<imports>
def plot(data: pd.DataFrame):
    chart = <stub>

    return chart;

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line.. """

        elif library == "altair":
            instructions = {
                "role": "system",
                "content": f"{general_instructions}. Always add a type that is BASED on semantic_type to each field such as :Q, :O, :N, :T, :G. Use :T if semantic_type is year or date. The plot method must return an altair object (chart)`. Think step by step. \n",
            }
            template = \
                """
# ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED. ALWAYS INCLUDE THE FUNCTION DEFINITION. 
import altair as alt
<imports>
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    return chart
chart = plot(data) # data already contains the data to be plotted.  Always include this line. No additional code beyond this line..
"""

        elif library == "plotly":
            instructions = {
                "role": "system",
                "content": f"{general_instructions} If calculating metrics such as mean, median, mode, etc. ALWAYS use the option 'numeric_only=True' when applicable and available, AVOID visualizations that require nbformat library. DO NOT inlcude fig.show(). The plot method must return an plotly figure object (fig)`. Think step by step. \n.",
            }
            template = \
                """
# ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED. ALWAYS INCLUDE THE FUNCTION DEFINITION. 
import plotly.express as px
<imports>
def plot(data: pd.DataFrame):
    fig = <stub> # only modify this section

    return chart
chart = plot(data) # variable data already contains the data to be plotted and should not be loaded again.  Always include this line. No additional code beyond this line..
"""

        else:
            raise ValueError(
                "Unsupported library. Choose from 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'ggplot', 'altair'."
            )

        return template, instructions
