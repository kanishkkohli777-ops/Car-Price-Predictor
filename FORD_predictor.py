import streamlit as st
import pandas as pd
import joblib

model = joblib.load('ford_price_prediction.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')


st.title("🚘CAR PRICE PREDICTOR🚗 ")
# st.image("https://th.bing.com/th/id/OIP.F0o4eCwFj09O0X0sqXo89AHaFv?w=233&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",width=100)




model_name=st.selectbox("Select Model",['Fiesta','Focus','Kuga','EcoSport','Puma'])



year=st.number_input("Year(2000-2026)",min_value=2000,max_value=2026,step=1)

mileage=st.slider("Mileage(km)",0,200000,100)

transmission=st.selectbox("Transmission Type",['Manual','Automatic','Semi-Auto'])

tax=st.number_input("Tax Range(0-200)",min_value=0,max_value=200)
FuelType=st.selectbox("FUEL TYPE",['Petrol','Diesel','Hybrid','Electric'])

mpg=st.number_input(" Enter Miles Per Gallon(mpg)(20-250)",min_value=20.0,max_value=250.0)

EngineSize=st.number_input("EngineSize(1.0-5.0)",min_value=1.0,max_value=5.0)



import streamlit as st

st.sidebar.title("🚗 Ford Car Price Predictor")

st.sidebar.info("""
This Model predicts the price of used Ford cars using Machine Learning.
""")

st.sidebar.markdown("""
### 📌 How to Use
1. Select car model  
2. Enter year  
3. Enter mileage  
4. Choose fuel type & transmission  
5. Enter tax, mpg, engine size  
6. Click Predict Price  

💡 Get instant estimated price
""")

st.sidebar.markdown("""
### ⚙️ Features
✔ Fast prediction  
✔ Easy to use  
✔ ML-based model  
✔ Accurate estimates  
""")

st.sidebar.warning("⚠️ Price is an estimate, not exact market value")


if st.button("Predict Price"):
    


    input_dic={
        "model": model_name,
        'year':year,
        'transmission':transmission,

        'FuelType':FuelType,
         'mileage':mileage,
         'mpg': mpg,
         'tax':tax,
         'engineSize':EngineSize

    }

    numeric_cols = ['year','mileage','tax','mpg','engineSize']
    cat_cols = [col for col in columns if col not in numeric_cols]
    input_df = pd.DataFrame([input_dic])

    scaled_numeric = scaler.transform(input_df[numeric_cols])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=cat_cols, fill_value=0)

    import numpy as np

    final_input = np.hstack([scaled_numeric, input_df_encoded.values])


    prediction = model.predict(final_input)
    st.success(f"Predicted price is $ {int(prediction[0]):,}")