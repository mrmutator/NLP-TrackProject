from annoy import AnnoyIndex

raw_input("<Enter> to create tree.")
tree = AnnoyIndex(500)

raw_input("<Enter> to load tree.")
tree.load("test_tree.ann")

raw_input("<Enter> to load 10,000 vectors.")
q = True
while q:

    for i in xrange(10000):
        tree.get_item_vector(i)

    resp = raw_input("<Enter> to load 10,000 vectors.")
    if resp.strip() == "q":
        q = False

raw_input("<Enter> to unload tree.")

tree.unload("test_tree.ann")

raw_input("done.")
tree.load("test_tree.ann")

raw_input("<Enter> to load 10,000 vectors.")
q = True
while q:

    for i in xrange(10000):
        tree.get_item_vector(i)

    resp = raw_input("<Enter> to load 10,000 vectors.")
    if resp.strip() == "q":
        q = False

raw_input("<Enter> to delete tree.")

del tree

raw_input("done.")
