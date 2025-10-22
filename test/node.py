from pyvis.network import Network

net = Network(notebook=True, directed=True)

net.add_node("Begin")
net.add_node("Retrieval")
net.add_node("Generate")
net.add_node("Answer")

net.add_edge("Begin", "Retrieval")
net.add_edge("Retrieval", "Generate")
net.add_edge("Generate", "Answer")

net.show("graph.html")