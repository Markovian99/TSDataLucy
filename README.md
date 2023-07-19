# Project Title: TimeSeriesData-Lucy

This Python application uses the power of GenAI to analyze .csv files containing time series data. It not only identifies insights, trends, and key findings but also provides an intuitive description of these discoveries. It produces detailed reports in a human-readable format, facilitating the understanding of complex time series data analysis.

## Features
- Process .csv files containing time series data
- Comprehensive analysis of insights, trends, and key findings
- Intuitive reports in natural language
- Easy setup and use

## Installation and Setup

### Prerequisites

You need to have Python 3.8 or later to use this application. If you don't have Python installed, you can download it from the official site: https://www.python.org/downloads/

### Steps

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/Markovian99/TSDataLucy.git
   ```
   
2. Navigate to the cloned project directory.

   ```bash
   cd TSDataLucy
   ```
   
3. Install the necessary packages using pip. (We recommend using a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```
   
4. In the project's root directory, create a `.env` file to store your API keys securely.
   
5. Open the `.env` file using any text editor and enter your API keys as shown below:

   ```bash
   export BARD_API_KEY = "YOUR BARD API KEY"
   export OPENAI_API_KEY = "YOUR OPENAI API KEY"
   ```
   If using Azure for OPEN AI, also include
   ```bash
   export OPENAI_API_BASE = "YOUR OPENAI API BASE"
   export OPENAI_API_TYPE = "YOUR OPENAI API TYPE"
   export OPENAI_API_VERSION = "YOUR OPENAI API VERSION"
   ```

## Usage

To run the TSDataLucy application:

```bash
cd src
streamlit run app.py
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the terms of the Apache License 2.0. For more details, please see the [LICENSE](LICENSE) file.

## Support

For any questions or issues, please contact the maintainers, or raise an issue in the GitHub repository.

Enjoy analyzing your time series data with TSDataLucy!
