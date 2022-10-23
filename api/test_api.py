import requests
url = "https://getaround-api-xgboost.herokuapp.com/predict"#"http://127.0.0.1:8000/predict"

# For local debugging, on a console, you need to run
# python -m uvicorn app:app --reload
# Here is an example input
res = requests.post(url, 
    json={
    "model_key": "Volkswagen",
    "mileage": 17200,
    "engine_power": 190,
    "fuel": "diesel",
    "paint_color": "brown",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
})

print("\n\testimated price is",res.json()) 
print("\n\tactual rental_price_per_day is 156 dollars per day\n")

# Here is an example  input with input issues
res = requests.post(url, 
    json={
    "model_key": "no car name",
    "mileage": 1000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
})

print("\n\testimated price is",res.json()) 
print("\n\tactual rental_price_per_day is 156 dollars per day\n")
