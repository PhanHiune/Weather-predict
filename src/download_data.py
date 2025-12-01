import pandas as pd
import requests

def download_historical():
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=16.0471&longitude=108.2068&"
        "start_date=2020-01-01&end_date=2024-12-31&"
        "hourly=temperature_2m,relativehumidity_2m,cloudcover,pressure_msl,rain,wind_speed_10m"
    )
    
    data = requests.get(url).json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "cloud": data["hourly"]["cloudcover"],
        "pressure": data["hourly"]["pressure_msl"],
        "rain": data["hourly"]["rain"],
        "wind": data["hourly"]["wind_speed_10m"]
    })

    df.to_csv("data/historical_weather.csv", index=False)
    print("Tải dữ liệu thành công → lưu vào data/historical_weather.csv")

if __name__ == "__main__":
    download_historical()
