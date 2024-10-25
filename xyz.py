import matplotlib.pyplot as plt
import networkx as nx

# Create a graph to visualize healthcare data sources
G = nx.Graph()

# Add nodes
G.add_node('Predictive Modeling', size=2000)
G.add_node('EHRs')
G.add_node('Clinical Trial Data')
G.add_node('Consumer-Generated Data')
G.add_node('Medical Imaging')

# Add edges
edges = [
    ('Predictive Modeling', 'EHRs'),
    ('Predictive Modeling', 'Clinical Trial Data'),
    ('Predictive Modeling', 'Consumer-Generated Data'),
    ('Predictive Modeling', 'Medical Imaging')
]
G.add_edges_from(edges)

# Set positions
pos = nx.spring_layout(G)

# Draw the network
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
plt.title('Healthcare Data Sources for Predictive Modeling')
plt.show()
