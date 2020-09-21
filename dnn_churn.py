# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Zhenhao Li
@description: DNN churn prediction
"""

import keras as K
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def dnn_churn(train_x, train_y, test_x, test_y):
	# model construction
	init = K.initializers.glorot_uniform(seed=1)
	simple_adam = K.optimizers.Adam()
	model = K.models.Sequential()
	model.add(K.layers.Dense(units=5, input_dim=66, kernel_initializer=init, activation='relu'))
	model.add(K.layers.Dropout(rate=0.2))
	model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
	model.add(K.layers.Dropout(rate=0.2))
	model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
	num_classes = 2
	train_y = K.utils.to_categorical(train_y, num_classes)
	test_y_copy = test_y.copy()
	test_y = K.utils.to_categorical(test_y, num_classes)

	# model training
	b_size = 32
	max_epochs = 25
	print("Starting training ")

	val_split = 0.2
	history = model.fit(train_x, train_y, validation_split=val_split, batch_size=b_size,
						epochs=max_epochs, shuffle=True, verbose=1)
	print("Training finished \n")
	print(model.summary())

	# model validation
	eval = model.evaluate(test_x, test_y, verbose=0)
	print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))
	mpr = model.predict_classes(test_x)

	# confusion matrix
	metrics.confusion_matrix(test_y_copy, mpr)

	# generate roc
	y_pred_proba = model.predict_proba(test_x, verbose=0)
	fpr, tpr, thresholds = roc_curve(test_y_copy, y_pred_proba[:, 1])
	roc_auc = auc(fpr, tpr)

	# model validation plot
	import matplotlib.pyplot as plt
	train_loss_history = history.history['loss']
	val_loss_history = history.history['val_loss']

	train_acc_history = history.history['accuracy']
	val_acc_history = history.history['val_accuracy']

	# outputFileFolder = "./data_" + method + "/"
	# if not os.path.exists(outputFileFolder):
	#     os.makedirs(outputFileFolder)

	# save data
	nx = len(train_loss_history)
	epochs = [i for i in range(len(train_loss_history))]

	# loss_vs_epoch = []
	# loss_vs_epoch.append(epochs)
	# loss_vs_epoch.append(train_loss_history)
	# loss_vs_epoch.append(val_loss_history)
	# np.savetxt(outputFileFolder + 'loss_vs_epoch' + '.csv', loss_vs_epoch, delimiter=',')

	# acc_vs_epoch = []
	# acc_vs_epoch.append(epochs)
	# acc_vs_epoch.append(train_acc_history)
	# acc_vs_epoch.append(val_acc_history)    
	# np.savetxt(outputFileFolder + 'acc_vs_epoch' + '.csv', acc_vs_epoch, delimiter=',')

	# # loss plot
	# plt.plot(epochs,train_loss_history, label='training loss')
	# plt.plot(epochs,val_loss_history, label='testing loss')
	#
	# plt.scatter(val_loss_history.index(min(val_loss_history)), min(val_loss_history),
	# 			c='r', marker='o', label='minimum_val_loss')
	# plt.xlabel("epoch")
	# plt.ylabel("loss")
	# plt.xticks(np.linspace(0, nx, 5))
	# plt.legend(loc='best')
	# plt.grid(True)
	# plt.title('training loss and testing loss vs. epoch number')
	# # plt.savefig(outputFileFolder + 'loss_vs_epoch.png')
	# plt.show()
	#
	# # accuracy plot
	# plt.plot(epochs,train_acc_history, label='training accuracy')
	# plt.plot(epochs,val_acc_history, label='testing accuracy')
	# plt.scatter(val_acc_history.index(max(val_acc_history)), max(val_acc_history),
	# 			c='r', marker='o', label='maximum_testing_accuracy')
	# plt.xlabel("epoch")
	# plt.ylabel("accuracy")
	# plt.xticks(np.linspace(0, nx, 5))
	# plt.legend(loc='best')
	# plt.grid(True)
	# plt.title('training accuracy and testing accuracy vs. epoch number')
	# # plt.savefig(outputFileFolder + 'accuracy_vs_epoch.png')
	# plt.show()

	return fpr, tpr, roc_auc

