problems with DL:
- Data Hungry
- computational heavy

pre-trained model help in this, but what if our classification is not present in the pre-trained model classes

### TRANSFER LEARNING:
- takes the convolutional part from the ppre-trained model and use it in our model
- we decide the fully connected layers
- train the model only on FC layers
- the convolutional layers are frozen to avoid training them
- implemented in two ways
	- Feature Extraction: Dense layers replaced. Used when the class to classified is already trained in the pre-trained model

	- Fine Tuning: Last few convolutional layers are also trained along with FC layers.Used when the class to identified is not present in the pre-trained model
