import matplotlib.pyplot as plt

x = ["JAN", "FEB", "MARCH", "APRIL", "MAY", "JUN", "JUL", "AUG"]
y = [2.8, 2, 4.3, 2.2, 8.2, 73.6,207.5,203.5]

# Plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label="Line Graph")

# Labels and Title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Matplotlib Graph")
plt.legend()

# Show the plot
plt.show()
