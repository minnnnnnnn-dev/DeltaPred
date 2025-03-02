import streamlit as st
import pandas as pd
from streamlit_plotly_events import plotly_events
from model_utils import load_model, predict
from utils import standardize_datetime
from plot_utils import plot_predictions, plot_updated_predictions

import asyncio
import sys

import plotly.graph_objects as go

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(
    layout='wide',
    page_title='Delta Pred'
)

def streamlit_main():
    """
    Streamlit Main Function
    """
    st.title('Delta Pred : Model Prediction & Interpretation')
    
    # Data Type (Tabular / Time Series)
    st.sidebar.header('Data Type')
    data_type = st.sidebar.radio('Select Data Type', ['Tabular', 'Time Series'])

    # DataFrame (Upload)
    st.sidebar.header('Data Upload')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Model (Upload), Pytorch/Tensorflow/Scikit-learn Supported (Tensorflow not implemented yet)
    st.sidebar.header("Upload Model File")
    model_file = st.sidebar.file_uploader("Choose a Model File", type=["pt", "pth", "pkl", "sav", "mth"]) # "h5", "keras"
    
    model, model_type = None, None
    
    if model_file:
        model, model_type = load_model(model_file)
        if model_type == "unsupported":
            st.sidebar.error("Unsupported Model File")
        else:
            st.sidebar.success(f"Loaded {model_type} Model Successfully")
    
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data")
        st.write(df.head())
        
        # Feature & Target Selection
        target_variable = st.selectbox("Select Target Variable", df.columns)
        input_features = [col for col in df.columns if col != target_variable]
        
        # DataFrame : Tabular
        if data_type == 'Tabular':
            id_column = st.selectbox("Select ID Column (if exists)", ["None"] + df.columns.tolist())
            if id_column != "None":
                df = df.drop(columns=[id_column])
                input_features = [col for col in input_features if col != id_column]
                
            # Make Prediction with Uploaded Model
            X_input = df[input_features].copy()
            original_predictions = predict(model, model_type, X_input, target_variable)
            df["Original Prediction"] = original_predictions
            
                
            ## PLOT 1 (True)
            fig1 = plot_predictions(df, target_variable=target_variable)
            st.plotly_chart(fig1, use_container_width=True)
            
            # selected_points = plotly_events(fig1) # Select Point, User wants to experiment with
            # selected_point_index = selected_points[0]['pointIndex'] if selected_points else None
            st.subheader("Select Data Point to Modify")
            min_index = 0
            max_index = len(df) - 1
            selected_point_index = st.number_input(
                "Enter the index of the data point to modify:",
                min_value=min_index,
                max_value=max_index,
                value=min_index,
                step=1
            )
            st.text(f"Selected Point Index : {selected_point_index}")
            
            if selected_point_index is not None:
                modified_df = X_input.copy()
                selected_row = modified_df.iloc[selected_point_index]
                
                # Add Slider for each selected feature
                modifiable_features = st.multiselect("Select Features to Modify", input_features)
                
                for feature in modifiable_features:
                    current_value = selected_row[feature] 
                    if current_value == 0:
                        min_value = -10.0
                        max_value = 10.0
                    else:
                        min_value = current_value - abs(current_value) * 0.5
                        max_value = current_value + abs(current_value) * 0.5

                    change = st.slider(
                        f"{feature} Change", 
                        min_value=min_value, 
                        max_value=max_value, 
                        value=current_value,  
                        key=f"slider_{feature}"
                    )

                    selected_row[feature] = change
                
                
                if st.button("Run Modified Prediction"):
                    modified_df.iloc[selected_point_index] = selected_row
                    new_predictions = predict(model, model_type, modified_df, target_variable)
                    df["Modified Prediction"] = new_predictions
                    
                    zoom_range = 20
                    zoom_start = max(0, selected_point_index - zoom_range)
                    zoom_end = min(len(df) - 1, selected_point_index + zoom_range)
                    xaxis_range = [df.index[zoom_start], df.index[zoom_end]] 
                    
                    fig2 = plot_updated_predictions(df, target_variable, xaxis_range)
                    st.plotly_chart(fig2, use_container_width=True)

                    

        
        # elif data_type == 'Time Series':
        #     # st.subheader("Time Series Prediction Settings")
            
        #     date_col = st.selectbox("Select Date Column", df.columns)
        #     df[date_col] = pd.to_datetime(df[date_col])
        #     df = df.set_index(date_col).sort_index()
            
        #     # Range Selection
        #     range_selection = st.select_slider("Select Input & Prediction Range", options=list(range(-30, 10)), value=(-10, 5))

        #     # `t-n` & `t+m` auto-selection
        #     time_window = range_selection[0]  # (t-n) : Input Window
        #     prediction_horizon = range_selection[1]  # (t+m) : Target

        #     st.text(f"Model Input Window: t{time_window} ~ t")  # 과거 입력 범위
        #     st.text(f"Prediction Target: t+{prediction_horizon}")  # 예측 목표      
            


        #     ## PLOT 1 (True)
        #     fig1 = go.Figure()
        #     fig1.add_trace(go.Scatter(
        #                 x = df.index, 
        #                 y=df[target_variable], 
        #                 mode="lines", 
        #                 name="True", 
        #                 line=dict(color="blue")
        #             )) # 실제값
            
        #     fig1.update_layout(
        #                 title="Actual Target Value (Time Series)",
        #                 xaxis=dict(title="Date", gridwidth=1, showgrid=True),
        #                 yaxis=dict(title="Target Value", gridwidth=1),
        #                 width=None,
        #                 paper_bgcolor='#0e1117',
        #                 plot_bgcolor='#262730',
        #                 font=dict(color='#FFFFFF', size=15),
        #                 xaxis_rangeslider_visible=True,
        #             )
        #     fig1.update_traces(showlegend=True)
            
        #     # Select Point, User wants to experiment with
        #     selected_points = plotly_events(fig1) # Point Selection Feature ([{'x': '2021-11-16 21:50', 'y': 1247.1, 'curveNumber': 0, 'pointNumber': 45818, 'pointIndex': 45818}])

            
        #     if len(selected_points) != 0:
        #         selected_point_x = selected_points[0]['x']
        #         selected_point_y = selected_points[0]['y']

        #         st.text('x : ' + selected_point_x)
        #         st.text('y : ' + str(selected_point_y))
                
        #         # To Standardize the datetime format
        #         formatted_date = standardize_datetime(selected_point_x)
                
        #         if formatted_date:
        #             ind_i = df.index.get_loc(formatted_date) if formatted_date in df.index else None
        #             if isinstance(ind_i, slice):
        #                 ind_i = ind_i.start
                    
        #             if ind_i is not None:
        #                 st.text(f"Index of Selected Point : {ind_i}")
        #             else:
        #                 st.error(("Selected Point not found in DataFrame Index"))
        #         else:
        #             st.error("Invalid Date Format")
        #     else:
        #         st.error("No Point Selected")




if __name__ == '__main__':
    streamlit_main()


