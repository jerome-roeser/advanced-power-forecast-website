
import streamlit as st
import requests

import numpy as np
import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import plotly.graph_objects as go


### API call ==================================================================
#base_url = "http://127.0.0.1:8000"
base_url = "https://power-v2-pdymu2v2na-ew.a.run.app"
#------------------------------------------------------------------------------

def call_visu(today_date):

    params_visu ={
        'input_date': today_date,   # '2000-05-15' (dt.date())
        'power_source': 'pv',
        'capacity': 'true'
        }

    endpoint_visu = "/visualisation"
    url_visu = f"{base_url}{endpoint_visu}"
    response_visu = requests.get(url_visu, params_visu).json()

    plot_df = pd.DataFrame.from_dict(response_visu)
    plot_df.utc_time = pd.to_datetime(plot_df.utc_time,utc=True)

    return plot_df

# Session states and Callbacks =================================================
# (see:https://docs.streamlit.io/library/advanced-features/button-behavior-and-examples)

def add_day():
    st.session_state['today'] += dt.timedelta(days=1)
    st.session_state['plot_df'] = call_visu( st.session_state['today'] )

def sub_day():
    st.session_state['today'] -= dt.timedelta(days=1)
    st.session_state['plot_df'] = call_visu( st.session_state['today'] )

# def calender(b):
#     st.session_state['plot_df'] = call_visu( b )


# initialize session states
if 'today' not in st.session_state:
    st.session_state['today'] = dt.date(2021, 7, 6) # default date

if 'plot_df' not in st.session_state:
    st.session_state['plot_df'] = call_visu( st.session_state['today'] )

# ==============================================================================
# ====================== Streamlit Interface ===================================

### Sidebar ====================================================================
st.sidebar.markdown(f"""
   # User Input
   """)


# Calender select
calender_today = st.sidebar.date_input(
                             label='Simulated today',
                             value= st.session_state['today'],
                             min_value=dt.date(2020, 1, 1),
                             max_value=dt.date(2022, 12, 30),
)

if st.session_state['today'] != calender_today:

    st.session_state['today'] = calender_today
    st.session_state['plot_df'] = call_visu( st.session_state['today'] )


# Move a day forth and back
columns = st.sidebar.columns(2)
columns[0].button('Day before', on_click=sub_day)
columns[1].button('Day after', on_click=add_day)

# Show values
show_true = st.sidebar.radio('Show true values', ('Yes', 'No'))



### Main window ====================================================================

f"""
# Day-Ahead Power Forecast
(v0.2)

Today is the **{st.session_state['today']}**. The Day-Ahead prediction is for the \
     **{st.session_state['today'] + pd.Timedelta(days=1)}**.

"""

### Show plots =================================================================
# used in the plots
today_date = st.session_state['today']
plot_df = st.session_state['plot_df']
#------------------------------------------------------------------------------

### capacity

# time variables
today_dt = pd.Timestamp(today_date, tz='UTC')
time = plot_df.utc_time.values

sep_future = today_dt + pd.Timedelta(days=1)
sep_past = today_dt
sep_order = today_dt + pd.Timedelta(hours=12)

# plot
fig, ax = plt.subplots(figsize=(15,5))

ax.axvline(sep_past, color='k', linewidth=0.7)
ax.axvline(sep_future, color='k', linewidth=0.7)
ax.vlines(sep_order, ymin=0, ymax=100, color='k', linewidth=0.7, linestyle='--')

# stats
alpha_stats = 0.2
ax.step(time, plot_df['min'].values, where='pre',
        color='k', linestyle=':', alpha=alpha_stats, label='min')
ax.step(time, plot_df['max'].values, where='pre',
        color='k', linestyle=':', alpha=alpha_stats, label='max')
ax.step(time, plot_df['mean'].values, where='pre',
        color='k', linestyle='-', alpha=alpha_stats, label='mean')

lower_bound = plot_df['mean'].values - 1 * plot_df['std'].values
upper_bound = plot_df['mean'].values + 1 * plot_df['std'].values
ax.fill_between(time, lower_bound, upper_bound, step='pre',
                color='gray',
                alpha=alpha_stats,
                label='std')

# true current production data
current = 37 # current production data
ax.step(time[:current], plot_df.cap_fac.values[:current], where='pre',
        color='orange', linewidth=4, label='true')

# prediction day ahead data
hori = -24
ax.step(time[hori:], plot_df.pred.values[hori:], where='pre',
        color='orange', linewidth=4, linestyle=':', label='pred')

###
if show_true == 'Yes':
    ax.step(time[-36:], plot_df.cap_fac.values[-36:], where='pre',
         color='orange', linewidth=4, linestyle='-', alpha=0.4)
    st.sidebar.write('')
else:
    st.sidebar.write('')

# date ticks
ax.xaxis.set_major_locator(dates.HourLocator(byhour=range(24), interval=12, tz='UTC'))
ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M %d/%m/%Y'))

ax.set_xlim(today_dt - pd.Timedelta(days=1), today_dt + pd.Timedelta(days=2))
ax.set_ylim(0,120.0)
ax.set_xlabel('Time')
ax.set_ylabel('Capacity factor in %')

ax.annotate('day-ahead',(0.77,0.9), xycoords='subfigure fraction')
ax.annotate('today',(0.48,0.9), xycoords='subfigure fraction')
ax.annotate('day-behind',(0.15,0.9), xycoords='subfigure fraction')
ax.annotate('order book closed',(0.51,0.77), xycoords='subfigure fraction')
ax.set_title(f"Day Ahead prediction for { sep_future.strftime('%d/%m/%Y') }")

ax.legend();

##
st.pyplot(fig)

### ============================================================================

#import plotly.graph_objects as go

# fig = go.Figure()

# # Stats
# alpha_stats = 0.2
# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'],
#         y=plot_df['min'],
#         mode='lines',
#         line=dict(color="black", width=1, dash="dot", shape='hv'),
#         name='min'
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'],
#         y=plot_df['max'],
#         mode='lines',
#         line=dict(color="black", width=1, dash="dot", shape='hv'),
#         name='max'
#     )
# )
# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'],
#         y=plot_df['mean'],
#         mode='lines',
#         line=dict(color="black", width=1, shape='hv'),
#         name='mean'
#     )
# )
# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'],
#         y=plot_df['mean'] - plot_df['std'],
#         mode='lines',
#         fill='tonexty',
#         fillcolor='rgba(128,128,128,0.2)',
#         line=dict(width=0, shape='hv'),
#         name='std'
#     )
# )
# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'][:37],  # Assuming 37 is the index where current production data ends
#         y=plot_df['cap_fac'][:37],
#         mode='lines',
#         line=dict(color="orange", width=3, shape='hv'),
#         name='true'
#     )
# )
# fig.add_trace(
#     go.Scatter(
#         x=plot_df['utc_time'][-24:],  # Assuming you're taking the last 24 hours for prediction
#         y=plot_df['pred'][-24:],
#         mode='lines', # mode='lines',
#         line=dict(color="orange", width=3, dash="dash", shape='hv'),
#         name='pred'
#     )
# )

# ###
# #fig.show()
# #st.pyplot(fig)
# st.plotly_chart(fig, use_container_width=True)


### electricity

# # time variables
# today_dt = pd.Timestamp(today_date, tz='UTC')
# time = plot_df.utc_time.values

# sep_future = today_dt + pd.Timedelta(days=1)
# sep_past = today_dt
# sep_order = today_dt - pd.Timedelta(hours=36)

# # plot
# fig, ax = plt.subplots(figsize=(15,5))

# ax.axvline(sep_past, color='k', linewidth=0.7)
# ax.axvline(sep_future, color='k', linewidth=0.7)
# ax.vlines(sep_order, ymin=0, ymax=100, color='k', linewidth=0.7, linestyle='--')

# # stats
# alpha_stats = 0.2
# ax.step(time, plot_df['min'].values, where='pre',
#         color='k', linestyle=':', alpha=alpha_stats, label='min')
# ax.step(time, plot_df['max'].values, where='pre',
#         color='k', linestyle=':', alpha=alpha_stats, label='max')
# ax.step(time, plot_df['mean'].values, where='pre',
#         color='k', linestyle='-', alpha=alpha_stats, label='mean')

# lower_bound = plot_df['mean'].values - 1 * plot_df['std'].values
# upper_bound = plot_df['mean'].values + 1 * plot_df['std'].values
# ax.fill_between(time, lower_bound, upper_bound, step='pre',
#                 color='gray',
#                 alpha=alpha_stats,
#                 label='std')

# # true
# current = 37 # current production data
# ax.step(time[:current], plot_df.electricity.values[:current], where='pre',
#         color='orange', linewidth=3, label='true')

# # prediction
# hori = -24
# ax.step(time[hori:], plot_df.pred.values[hori:], where='pre',
#         color='orange', linewidth=3, linestyle=':', label='pred')

# # date ticks
# ax.xaxis.set_major_locator(dates.HourLocator(byhour=range(24), interval=12, tz='UTC'))
# ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M %d/%m/%Y'))

# #ax.set_xlim(today_dt - pd.Timedelta(days=2), today_dt + pd.Timedelta(days=1))
# ax.set_xlim(today_dt - pd.Timedelta(days=1), today_dt + pd.Timedelta(days=2))
# ax.set_ylim(0,1.0)
# ax.set_xlabel('Time')
# #ax.set_ylabel('Capacity factor in %')

# ax.annotate('day-ahead',(0.77,0.9), xycoords='subfigure fraction')
# ax.annotate('today',(0.48,0.9), xycoords='subfigure fraction')
# ax.annotate('day-behind',(0.15,0.9), xycoords='subfigure fraction')
# ax.annotate('order book closed',(0.51,0.77), xycoords='subfigure fraction')

# ax.legend();

# ##
# st.pyplot(fig)






# # model
# params_model ={
#     'input_date':input_prediction_date,
#     'n_days': 2,
#     'power_source': 'pv'
#     }

# endpoint_model = "/predict"
# url_model= f"{base_url}{endpoint_model}"
# response_model = requests.get(url_model, params_model).json()


# # baseline
# params_baseline ={
#     'input_date':input_prediction_date,
#     'n_days': 2,
#     'power_source': 'pv'
#     }

# endpoint_baseline = "/baseline_yesterday"
# url_baseline= f"{base_url}{endpoint_baseline}"
# response_baseline = requests.get(url_baseline, params_baseline).json()

# # data
# params_data ={
#     'input_date':input_prediction_date,
#     'n_days': 10,
#     'power_source': 'pv'
#     }

# endpoint_data = "/extract_data"
# url_data = f"{base_url}{endpoint_data}"
# response_data = requests.get(url_data, params_data).json()


# Main Panel
# Write name of chosen location
#st.write(f"**Chosen location:** :red[{location}]")


# # set-up 4 DatFrames according to input date and type of model
# X = pd.DataFrame(response_data.get(input_prediction_date)['days_before'])
# y = pd.DataFrame(response_data.get(input_prediction_date)['day_after'])
# y_baseline = pd.DataFrame(response_baseline.get(input_prediction_date))
# y_predicted = pd.DataFrame(response_model.get('dataframe to predict'))

# # convert date columns to datetime object
# X.date = pd.to_datetime(X.date,utc=True)
# y.date = pd.to_datetime(y.date, utc=True)
# y_baseline.date = pd.to_datetime(y_baseline.date, utc=True) + datetime.timedelta(days=1)

# # Matplotlib pyplot of the PV data
# fig, ax = plt.subplots(sharex=True, sharey=True)
# ax.plot(X.date, X.power_source, label='current production data')
# ax.plot(y.date, y.power_source, label='true production')
# ax.plot(y_baseline.date, y_baseline.power_source, label='baseline estimate')
# plt.ylim(0,1)
# plt.legend()
# st.pyplot(fig)


# # Metrics
# mean_training = X.power_source.mean()
# mean_baseline = y_baseline.power_source.mean()
# mean_diff = mean_baseline - mean_training

# # Trick to use 4 columns to display the metrics centered below graph
# col1, col2, col3, col4 = st.columns(4)
# col2.metric("Training", round(mean_training,3), "")
# col3.metric("Predicted", round(mean_baseline,3), round(mean_diff,3))


# # Map with the location
# coordinates = {
#             'Berlin - Tempelhof':{'lat':52.4821,'lon':13.3892},
#             'Berlin - Tegel':{'lat':52.5541,'lon':13.2931},
#             'Berlin - Schönefeld':{'lat':52.3733,'lon':5064},
#                }

# map =folium.Map(
#     location=[coordinates[location]['lat'],
#               coordinates[location]['lon']],
#     zoom_start=13)
# folium.Marker([coordinates[location]['lat'], coordinates[location]['lon']],
#               popup=location,
#               icon=folium.Icon(color='red')).add_to(map)
# folium_static(map)
