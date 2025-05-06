# Genetic Algorithm Visualizer

A modern, interactive visualization of a genetic algorithm finding the minimum of a mathematical function. This project demonstrates how genetic algorithms work through an intuitive web interface with real-time visualizations.

## Features

- Interactive web-based UI using Streamlit
- Real-time visualization of the genetic algorithm's progress
- 3D surface plot of the objective function
- Dynamic parameter adjustment
- Progress tracking and results display
- Modern, responsive design

## Objective Function

The algorithm aims to find the minimum of the function:
```
f(x,y) = x² - 5y + 31
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arminmehraeen/Genetic-Algorithm.git
cd Genetic-Algorithm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to adjust parameters:
   - Set the number of generations
   - Click "Run Algorithm" to start the optimization

4. View the results:
   - Progress chart showing fitness scores over generations
   - 3D surface plot of the objective function
   - Table of top 10 solutions
   - Best solution found

## How It Works

1. **Initialization**: Creates a random population of solutions
2. **Fitness Evaluation**: Calculates how good each solution is
3. **Selection**: Keeps the best solutions
4. **Reproduction**: Creates new solutions based on the best ones
5. **Mutation**: Adds small random changes to maintain diversity
6. **Iteration**: Repeats the process until convergence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Armin Mehraeen

## Acknowledgments

- Streamlit for the web framework
- Plotly for interactive visualizations
- NumPy for numerical computations

