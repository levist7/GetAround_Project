import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import requests
import re
import plotly.express as px
import plotly.graph_objects as go

# Config
st.set_page_config(
    page_title="Getaround Delay Analysis 🚗",
    page_icon="⌛ ⏱️ 🚗 🚙",
    layout="wide"
)
# App
st.title('Getaround Delay Analysis - Dashboard ⏱️ 🚗')
# Loading Image and Text
image = Image.open('getaround_image.png')
col1, col2, col3 = st.columns([1.5, 5, 1.5])
col2.image(image, caption='Getaround user in action (Credit: Getaround.com)')
st.markdown("""
    :wave: Hello there and welcome to this dashboard!
    When using Getaround, drivers book cars for a specific time period: hours to days.
    Users need to bring back the car on time.
    It happens from time to time that drivers are late for the checkout.
    Late returns at checkout may generate problems for the next driver, especially, if the car is reserved on the same day.
    It results in negative feedback from customers as they have to wait for the car returning back. Some even canceled their rental.
    :dart: The goal of this dashboard is to give some hints on the impact of introducing time threshold on rentals.
    Within the time threshold, a car will not be displayed in the search results if the requested checkin or checkout times are very close.
    🧐 In this dashboard, you can sail down the historical dataset of getaround usage. The goal is to give some hints on the trade-off of minimum delay and its impact on usage and revenue.
    By examining historical data below, it can kick off the discussions on questions below:
    - How often are drivers late for the next check-in? How does it impact the next driver?
    - Which share of our owner’s revenue would potentially be affected by the feature?
    - How many rentals would be affected by the feature depending on the threshold and scope we choose?
    - How many problematic cases will it solve depending on the chosen threshold and scope?
    🚀 Let's start
""")
st.markdown("---")


@st.cache(allow_output_mutation=True)
def load_data():
    fname = 'get_around_delay_analysis_clean.csv'
    data = pd.read_csv(fname)
    return data


st.text('Loading data...')
dataset2 = load_data()


# Content
st.sidebar.header("Table of content")
st.sidebar.markdown("""
    * [Preview of data set](#dataset-preview)
    * [Plot 1](#plot-1) - Distribution of rentals being on time or late
    * [Plot 2](#plot-2) - Distribution of rentals being on time or late by their status
    * [Plot 3](#plot-3) - Distribution of rentals being on time or late by their status and checkin types
    * [Plot 4](#plot-4) - A different way of plotting the previous figure
    * [Plot 5](#plot-5) - Distribution of delta time between two rentals
    * [Plot 6](#plot-6) - Correlation between features
    * [Conclusions](#conclusions)
    * [Any solutions?](#any-solutions?)
""")

st.markdown("---")
st.subheader('Dataset Preview')
# Run the below code if the check is checked
if st.checkbox('Show processed data'):
    st.subheader('Overview of 10 random rows')
    st.write(dataset2.sample(10))
st.markdown("""
    In this dataset, some possible outliers have been avoided.
    Cleaned delay dataset has 20 980 rows. 330 rows were suspected as outliers and were removed.
    Also, factorized versions of non-numerical features were added for various reasons.
    Please see my github repo below for more details.
""")

# Plot 1
st.subheader('Plot 1')
st.markdown("Distribution of rentals being on time or late")
fig1 = px.histogram(dataset2, x='delay')
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
    We can see that delay at checkout is very common among Getaround drivers.
    It can range from couple minues to more than an hour.
""")
st.markdown("---")

# Plot 2
st.subheader('Plot 2')
st.markdown("Distribution of rentals being on time or late by their status")
fig2 = px.histogram(dataset2, x='state', color='delay')
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
    Around 3 200 Getaround users canceled their ride possibly due to the delay at checkout.
""")
st.markdown("---")

# Plot 3
st.subheader('Plot 3')
st.markdown(
    "Distribution of rentals being on time or late by their status and checkin types")
fig3 = px.histogram(dataset2, x='state', color='delay',
                    facet_col='checkin_type')
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""
    Please check the next figure for the comments.
""")
st.markdown("---")

# Plot 4
st.subheader('Plot 4')
st.markdown(
    "Playing with parameters to be able to interpret previous plot a bit better.")
fig4 = px.histogram(dataset2, x='delay', color='state',
                    facet_col='checkin_type')
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
    Checkin type of 'connect' are much less used by drivers than traditional mobile way of checking in.
    The drivers who checked in with 'connect' feature had much less delay in proportion than the drivers who checked in without 'connect' feature.

    It seems like the checkin type of 'connect'
    * reduces the late checkouts and
    * is likely to reduce frictions among Getaround drivers.
""")
st.markdown("---")

# Plot 5
st.subheader('Plot 5')
st.markdown("Distribution of delta time between two rentals in minutes")
fig5 = px.box(
    dataset2,
    x='state',
    y='time_delta_with_previous_rental_in_minutes',
    facet_col='checkin_type')
fig5.update_layout(yaxis_title="Delta time between two rentals in minutes")
# quartilemethod="exclusive") # or "inclusive", or "linear" by default
st.plotly_chart(fig5, use_container_width=True)
("""
    Delta time between two rentals in minutes does not seem to have an obvious impact.
""")
st.markdown("---")

# Plot 6
st.subheader('Plot 6')
st.markdown("Correlation between features")
corr_match = dataset2.corr().loc[:, ['factorized_delay']].abs(
).sort_values(by='factorized_delay', ascending=False)
key_cols = corr_match.index[:-3]  # sorted column names
df_corr = dataset2[key_cols].corr().abs()
fig6 = go.Figure()
fig6.add_trace(
    go.Heatmap(
        x=df_corr.columns,
        y=df_corr.index,
        z=np.array(df_corr)
    )
)
st.plotly_chart(fig6, use_container_width=True)
("""
    - Delay at checkout is correlated to check-in type.
    - There is colinearity between whether user canceled the ride or not and delay at checkout.
    - Time_delta_with_previous_rental does not seem to have a collinearity.
""")
st.markdown("---")


# Conclusion

st.subheader('Conclusions')
# some calculations below
tot_data = dataset2.shape[0]  # number of cases
mask_0 = dataset2.state == "canceled"
tot_cancel = mask_0.sum()  # total number of canceled cases
tot_cancel_percent = round(100. * (tot_cancel / tot_data), 1)  # percent
tot_ended = tot_data - tot_cancel  # total number of ended cases
col_ = "delay_at_checkout_in_minutes"
mask_a = dataset2[col_] >= 0.0
tot_delay_today = mask_a.sum()  # total number of delayed cases
tot_delay_today_percent = round(
    100. * (tot_delay_today / tot_ended), 1)  # percentage

st.write("1. ", tot_delay_today, " (", tot_delay_today_percent,
         "percent) drivers were late for the next check-in.")
st.write("2. It possibly resulted in ", tot_cancel, " (", tot_cancel_percent,
         " percent) users to cancel their rental requests.")

st.markdown("""
    Late arrivals increase the chances of a ride being canceled and so company's losing money. It increases the risk on financial side.
    It is highly recommended to optimize the financial risk by introducing a threshold on delay.
""")
input_threshold = st.slider(
    '3. Move the slider of delay threshold to see its impact',
    0,
    150,
    step=5)

try:  # in case user has interacted with the slider
    mask_b = dataset2[col_] >= input_threshold
except BaseException:  # in case user has not interacted with the slider yet
    mask_b = mask_a
tot_delay_tomorrow = mask_b.sum()  # number of delays after introducing threshold
# number of delay issues resolved
change_delay = tot_delay_today - tot_delay_tomorrow
change_delay_percent = round(100. * change_delay / tot_ended, 1)

st.write("\n\t :star: If the delay threshold above had been introduced, ",
         change_delay, " (", change_delay_percent,
         " percent) problematic cases would have been resolved.\n")

type_checkin = st.selectbox(
    '4. Select the feature on checkin type', [
        "Only connect", "No change"])
mask_c = dataset2["checkin_type"] == "connect"
total_delay_connect = (mask_a & mask_c).sum()
total_delay_connect_percent = round(
    100. * (total_delay_connect / tot_delay_today), 1)
tot_delay_tomorrow_percent = tot_delay_today_percent - total_delay_connect_percent
if type_checkin == "Only connect":
    st.write(
        "\t:star:With the use of checkin type above, ",
        tot_delay_tomorrow_percent,
        " percentage of problematic cases would have been resolved. \n")
else:
    st.write("\t:star:Sorry, the use of above feature does not resolve any issues.")
st.markdown("---")


# Plot 7
st.subheader('Any solutions?')
# A threshold sampled at each 5 mins for a day
threshold = np.arange(0, 60 * 24, 2.5)
rent_permin = 119. / 24. / 60.  # median rental price $ per min
rental_duration = 4 * 60.  # use of 4 hours is estimated for all cancelled rides
col_ = 'delay_at_checkout_in_minutes'
risk_percentage = []
for val_delay in threshold:
    mask_0 = dataset2[col_] > val_delay
    tot_late_mins = dataset2.loc[mask_0, col_].sum()
    earn_late = rent_permin * tot_late_mins
    count_late = mask_0.sum()
    risk_late = count_late * rent_permin * rental_duration
    risk_percentage.append(risk_late / earn_late * 100.)  # in percentage

st.markdown("Figure on threshold time to disable car listing versus metric used to quantify the risk of losing money")
fig7 = px.line(x=threshold, y=risk_percentage)
fig7.add_hline(y=100., line_color="red")
fig7.update_layout(
    xaxis_title='Threshold in minutes',
    yaxis_title='Metric used to quantify the risk of losing money')
st.plotly_chart(fig7, use_container_width=True)

st.markdown("""
    Late checkouts cause frictions among drivers and pose a risk on financial side.
    For this purpose, a metric is defined to quantify the risk of losing money as calculated below
    1. Calculate the amount of money risked due to late arrivals for each threshold delay
    2. Calculate the amount of money earned by supplementary minutes by late arrivals.
    3. Calculate the metric = risked amount of money / money earned by late arrivals
    Without any thresholds, the company takes a risk of losing a lot of money.
    At current situation, calculated metric is 193%.
	It signifies that the company could have lost two times more money than they had earned from users returning cars late.
    It is an important risk that the company had taken.
	I assume the worst case scenario where all the late checkouts lead a cancelation of a new ride of 4 hour long.
    It is neccessary to find an optimal threshold level on delay to reduce the financial risk.
    In the worst case scenario, **at 60 minutes**, the money earned by late arrivals is equal to the money risked due to the delayed checkouts.
	*It would serve the purpose very well to do an A/B testing to optimize this threshold on delay before applying it to its whole network.*
    For the calculations, I assume
    * a rental price per day of 119 dollars (obtained from Getaround dataset) and
    * a rental duration of 4 hours (personal judgement).
    For more details, checkout my github repo.
""")
st.markdown("---")

# Footer
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        🍇 🍇 🍇 🍇 🍇 🍇
        If you want to learn more, check out [my Github](www.github.com/levist7)
    """)
