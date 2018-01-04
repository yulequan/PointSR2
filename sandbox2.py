import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

input = batch_data_input[0]
ax.scatter(input[:,0], input[:,1], input[:,2], c='g', marker='.')
input2 = batch_data_input[point_order[:,:512,0],point_order[:,:512,1]][0]
ax.scatter(input2[:,0], input2[:,1], input2[:,2], c='b', marker='o')
