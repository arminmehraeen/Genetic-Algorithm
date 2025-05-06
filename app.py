import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import random
import sys
from typing import List, Tuple

# Constants
N = 2000  # Population size
M = 100   # Number of best solutions to keep

def objective_function(x: float, y: float) -> float:
    """Objective function to minimize."""
    return (x ** 2) - (y * 5) + 31

def fitness(x: float, y: float) -> float:
    """Calculate fitness score for a solution."""
    ans = objective_function(x, y)
    if ans == 0:
        return sys.maxsize
    return abs(1 / ans)

def create_initial_population() -> List[Tuple[float, float]]:
    """Create initial random population."""
    return [(random.uniform(-M, M), random.uniform(-M, M)) for _ in range(N)]

def genetic_algorithm(iterations: int, population_size: int, mutation_rate: float) -> List[Tuple[float, Tuple[float, float]]]:
    """Run genetic algorithm for specified number of iterations."""
    solutions = [(random.uniform(-M, M), random.uniform(-M, M)) for _ in range(population_size)]
    best_solutions = []
    
    for i in range(iterations):
        # Rank solutions
        ranked_solutions = [(fitness(s[0], s[1]), s) for s in solutions]
        ranked_solutions.sort(reverse=True)
        
        # Store best solution
        best_solutions.append(ranked_solutions[0])
        best_solutions_current = ranked_solutions[:M]
        
        # Create new generation
        elements = []
        for s in best_solutions_current:
            elements.extend([s[1][0], s[1][1]])
        
        new_gen = []
        for _ in range(population_size):
            x = random.choice(elements) * random.uniform(1 - mutation_rate, 1 + mutation_rate)
            y = random.choice(elements) * random.uniform(1 - mutation_rate, 1 + mutation_rate)
            new_gen.append((x, y))
        
        solutions = new_gen
    
    return best_solutions

def create_3d_surface():
    """Create 3D surface plot of the objective function."""
    x = np.linspace(-M, M, 100)
    y = np.linspace(-M, M, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title='Objective Function Surface',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x,y)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def create_contour_plot(best_solutions: List[Tuple[float, Tuple[float, float]]]):
    """Create contour plot with solution path."""
    x = np.linspace(-M, M, 100)
    y = np.linspace(-M, M, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)
    
    # Extract solution coordinates
    solution_x = [solution[1][0] for solution in best_solutions]
    solution_y = [solution[1][1] for solution in best_solutions]
    
    fig = go.Figure(data=[
        go.Contour(x=x, y=y, z=Z, colorscale='Viridis', showscale=True),
        go.Scatter(
            x=solution_x,
            y=solution_y,
            mode='lines+markers',
            name='Solution Path',
            line=dict(color='red'),
            marker=dict(size=8)
        )
    ])
    
    fig.update_layout(
        title='Solution Path on Contour Plot',
        xaxis_title='X',
        yaxis_title='Y',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def main():
    st.set_page_config(page_title="Genetic Algorithm Visualizer", layout="wide")
    
    # Title and Introduction
    st.title("Genetic Algorithm Visualizer")
    
    # Introduction with expandable details
    with st.expander("About this Application", expanded=True):
        st.markdown("""
        This application demonstrates a genetic algorithm finding the minimum of the function:
        ```
        f(x,y) = x² - 5y + 31
        ```
        
        ### How Genetic Algorithms Work:
        1. **Initialization**: Creates a random population of solutions
        2. **Fitness Evaluation**: Calculates how good each solution is
        3. **Selection**: Keeps the best solutions
        4. **Reproduction**: Creates new solutions based on the best ones
        5. **Mutation**: Adds small random changes to maintain diversity
        6. **Iteration**: Repeats the process until convergence
        
        ### Understanding the Visualizations:
        - **3D Surface Plot**: Shows the objective function's landscape
        - **Contour Plot**: Displays the solution path on a 2D view
        - **Progress Chart**: Shows how the fitness score improves over generations
        - **Results Table**: Lists the best solutions found
        """)
    
    # Sidebar controls with more parameters
    st.sidebar.header("Algorithm Parameters")
    
    with st.sidebar.expander("Advanced Settings", expanded=True):
        iterations = st.slider("Number of Generations", 10, 200, 50)
        population_size = st.slider("Population Size", 100, 5000, 2000)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.2, 0.01, 0.01)
    
    if st.sidebar.button("Run Algorithm", type="primary"):
        with st.spinner("Running genetic algorithm..."):
            # Run algorithm
            best_solutions = genetic_algorithm(iterations, population_size, mutation_rate)
            
            # Create progress chart
            progress_data = pd.DataFrame([
                {
                    "Generation": i,
                    "Best Fitness": score,
                    "X": solution[0],
                    "Y": solution[1]
                }
                for i, (score, solution) in enumerate(best_solutions)
            ])
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Progress", "Visualizations", "Results"])
            
            with tab1:
                st.subheader("Optimization Progress")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_progress = go.Figure()
                    fig_progress.add_trace(go.Scatter(
                        x=progress_data["Generation"],
                        y=progress_data["Best Fitness"],
                        mode='lines+markers',
                        name='Fitness Score'
                    ))
                    fig_progress.update_layout(
                        title="Fitness Score Over Generations",
                        xaxis_title="Generation",
                        yaxis_title="Fitness Score",
                        showlegend=True
                    )
                    st.plotly_chart(fig_progress, use_container_width=True)
                
                with col2:
                    st.metric(
                        "Best Fitness Score",
                        f"{round(best_solutions[-1][0], 5)}",
                        f"{round(best_solutions[-1][0] - best_solutions[0][0], 5)}"
                    )
                    st.metric(
                        "Improvement",
                        f"{round((best_solutions[-1][0] - best_solutions[0][0]) / best_solutions[0][0] * 100, 2)}%"
                    )
            
            with tab2:
                st.subheader("Solution Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_3d_surface(), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_contour_plot(best_solutions), use_container_width=True)
            
            with tab3:
                st.subheader("Top Solutions")
                results_df = pd.DataFrame([
                    {
                        "Rank": i+1,
                        "X": round(solution[0], 5),
                        "Y": round(solution[1], 5),
                        "Fitness": round(score, 5),
                        "Objective Value": round(objective_function(solution[0], solution[1]), 5)
                    }
                    for i, (score, solution) in enumerate(best_solutions[:10])
                ])
                st.dataframe(results_df, use_container_width=True)
                
                # Display final objective value
                best_score, best_solution = best_solutions[-1]
                st.success(f"""
                Best solution found:
                - X: {round(best_solution[0], 5)}
                - Y: {round(best_solution[1], 5)}
                - Objective Value: {round(objective_function(best_solution[0], best_solution[1]), 5)}
                - Fitness Score: {round(best_score, 5)}
                """)

if __name__ == "__main__":
    main() 