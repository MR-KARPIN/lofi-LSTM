import midi2img
import os

# assign directory
directory = 'dataset'

# iterate over files in
# that directory
for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	# checking if it is a file
	if os.path.isfile(f) and f.endswith('.mid'):
		print("imagenificamos " + filename)
		midi2img.midi2image(os.path.join(directory, filename))





