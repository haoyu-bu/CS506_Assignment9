import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        # Initialize weights and biases
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        # Store intermediate activations for visualization
        self.a_hidden = None
        self.a_output = None
        self.gradients = {}

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def forward(self, X):
        self.z_hidden = X @ self.w1 + self.b1
        self.a_hidden = self.activation(self.z_hidden)
        self.z_output = self.a_hidden @ self.w2 + self.b2
        self.a_output = self.activation(self.z_output)
        return self.a_output

    def backward(self, X, y):
        # Compute loss gradient w.r.t. output
        error = self.a_output - y
        d_output = error * self.activation_derivative(self.z_output)

        # Gradients for hidden-to-output weights
        self.gradients['dw2'] = self.a_hidden.T @ d_output
        self.gradients['db2'] = np.sum(d_output, axis=0, keepdims=True)

        # Gradients for input-to-hidden weights
        d_hidden = (d_output @ self.w2.T) * self.activation_derivative(self.z_hidden)
        self.gradients['dw1'] = X.T @ d_hidden
        self.gradients['db1'] = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights and biases
        self.w1 -= self.lr * self.gradients['dw1']
        self.b1 -= self.lr * self.gradients['db1']
        self.w2 -= self.lr * self.gradients['dw2']
        self.b2 -= self.lr * self.gradients['db2']

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    for _ in range(10):  # Perform multiple training steps per frame
        mlp.forward(X)
        mlp.backward(X, y)

    # Clear axes for redrawing
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Step number for titles
    step_number = frame * 10
    ax_hidden.set_title(f"Hidden Space at Step {step_number}")
    ax_input.set_title(f"Input Space at Step {step_number}")
    ax_gradient.set_title(f"Gradient Visualization at Step {step_number}")

    # Hidden layer visualization
    hidden_features = mlp.a_hidden
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_xlabel("Hidden 1")
    ax_hidden.set_ylabel("Hidden 2")
    ax_hidden.set_zlabel("Hidden 3")

    # Add hyperplane (decision boundary) in hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    zz = (-mlp.w2[0, 0] * xx - mlp.w2[1, 0] * yy - mlp.b2[0, 0]) / mlp.w2[2, 0]
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='tan')

    # Input space decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], alpha=0.5, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_xlabel("x1")
    ax_input.set_ylabel("x2")

    # Gradient visualization
    input_nodes = ["x1", "x2"]
    hidden_nodes = ["h1", "h2", "h3"]
    output_node = "y"

    node_positions = {
        "x1": (0, 0.3), "x2": (0, -0.3),  # Input nodes
        "h1": (1, 0.75), "h2": (1, -0.75), "h3": (1, 0),  # Hidden nodes
        "y": (2, 0),  # Output node
    }

    scaling_factor = 0.01  # Adjust this factor to control the line width scaling
    min_linewidth = 1    # Set a minimum width for better visibility

    # Plot edges from input to hidden layer
    for i, input_node in enumerate(input_nodes):
        x1, y1 = node_positions[input_node]
        for j, hidden_node in enumerate(hidden_nodes):
            x2, y2 = node_positions[hidden_node]
            gradient_value = np.abs(mlp.w1[i, j])  # Use correct weight index
            line_width = max(gradient_value * scaling_factor, min_linewidth)  # Scale and set a minimum width
            ax_gradient.plot([x1, x2], [y1, y2], color="purple", linewidth=line_width, alpha=0.7)
            ax_gradient.text((x1 + x2) / 2, (y1 + y2) / 2, f"{gradient_value:.2f}", color="black", fontsize=8)

    # Plot edges from hidden layer to output
    for j, hidden_node in enumerate(hidden_nodes):
        x1, y1 = node_positions[hidden_node]
        x2, y2 = node_positions[output_node]
        gradient_value = np.abs(mlp.w2[j, 0])  # Use correct weight index
        line_width = max(gradient_value * scaling_factor, min_linewidth)  # Scale and set a minimum width
        ax_gradient.plot([x1, x2], [y1, y2], color="purple", linewidth=line_width, alpha=0.7)
        ax_gradient.text((x1 + x2) / 2, (y1 + y2) / 2, f"{gradient_value:.2f}", color="black", fontsize=8)

    # Add nodes
    for node, (x, y) in node_positions.items():
        ax_gradient.add_patch(Circle((x, y), radius=0.05, color="blue"))
        ax_gradient.text(x, y + 0.1, node, color="black", fontsize=10, ha="center")
    
    ax_gradient.set_xlim(-1, 3)
    ax_gradient.set_ylim(-1, 1)
    ax_gradient.axis("off")



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)