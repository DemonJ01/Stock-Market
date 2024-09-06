# Stock-Market-Trend-Analysis-using-LSTM-Model

## Overview
This project uses Long Short-Term Memory (LSTM) networks to predict stock prices. It includes the implementation of an LSTM model to analyze historical stock data and make predictions for future closing prices. The project covers the training of the model, evaluation metrics, and visualization of the predictions against actual stock prices.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: numpy, pandas, matplotlib, scikit-learn, keras, yfinance

USAGE----
When prompted, enter the stock name for which you want to predict prices.
The script downloads historical stock data using Yahoo Finance API.
It visualizes the stock's closing prices along with Nifty 50 index.
The LSTM model is trained and evaluated using mean absolute error, mean absolute percentage error, and median absolute percentage error.
Predictions are visualized against the actual stock prices.

Results
The script generates plots showing the stock's historical prices, predicted prices, and the difference between them.
Evaluation metrics (MAE, MAPE, MDAPE) are displayed in the console.
The last known price and the predicted price for the next day are printed.

output pics :
1)  [image](https://github.com/paradisekin/Stock-Market-Trend-Analysis-using-LSTM-Model/assets/126254105/da911c68-fec9-4ec9-92c8-31dfabf11ad0)
2)  [image](https://github.com/paradisekin/Stock-Market-Trend-Analysis-using-LSTM-Model/assets/126254105/8b97b756-5b05-4ac9-a93a-39d0cd0c2921)
3)  [image](https://github.com/paradisekin/Stock-Market-Trend-Analysis-using-LSTM-Model/assets/126254105/2e2defbe-ca06-4959-8e9e-52e1e7b7e9f1)


Running the Project in VS Code-----

1.Clone or Download the Repository:

2.Copy or fork the repository from GitHub.
If you prefer, you can download the code as a ZIP file and extract it.
Open the Project in VS Code:

3.Open Visual Studio Code on your local machine.
Install Required Libraries:

4.Open the terminal in VS Code.
Run the following command to install the necessary libraries:   pip install -r requirements.txt

5.Enter Stock Name:
When prompted, enter the name of the stock for which you want to predict future prices.

6.View Results:
The script will download data, train the LSTM model, and display visualizations and evaluation metrics.


Running the Project in Google Colab
1.Open Google Colab.

2.Create a New Notebook:
Click on "File" > "New Notebook" in Google Colab.

3.Copy and Paste Code:
Copy the entire content of stock_prediction.py script.
Paste it into a code cell in the Colab notebook.

4.Run the Notebook:
Execute the code cell.
Follow the prompts to enter the stock name for prediction.

5.View Results:
Colab will display visualizations and results directly in the notebook.
No need to install additional libraries as Colab comes with them preinstalled.


Notes:
Google Colab Recommended:
Using Google Colab is recommended as it eliminates the need to install extra libraries locally.

Internet Connection:
Ensure your machine or Colab notebook has an internet connection for downloading stock data.

Feedback and Issues:
Feel free to provide feedback issues on the GitHub repository.






