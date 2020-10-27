from tkinter import *
from tkinter import filedialog, messagebox as mb
from PIL import Image, ImageTk
from imageai.Prediction import ImagePrediction
import os

# path to execution file
exec_path = os.getcwd()
# default image path
path_to_image = exec_path + "\images\pizza8.jpg"
# default info&result text
info_text_default = "Here is a little tip!\n1. Choose an image\n2. Click \'Analyse\'\n3. See the result!"


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Strong pizza classifier")
        self.root.resizable(width=False, height=False)
        self.root.geometry('640x480')

        # loading default image
        self.image = Image.open(path_to_image)
        self.image = self.image.resize((300, 300), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)

        # creating default text with information
        self.label_info = Label(master=self.root, font="Calibri 13", justify=LEFT, text=info_text_default)
        self.label_info.place(x=0, y=0)

        # adding Analyse button
        self.button_start = Button(self.root, text="Analyse", command=self.analyse, padx=20, pady=5)
        self.button_start.place(relx=.5, rely=.8, anchor='s', height=30, width=130)

        # adding button for choosing images
        self.button_choose = Button(self.root, text="Choose image", command=self.choose_image, padx=20, pady=5)
        self.button_choose.place(relx=.5, rely=.87, anchor='s', height=30, width=130)

        # adding exit button
        self.button_exit = Button(self.root, text="Exit", command=self.root.quit, padx=20, pady=5)
        self.button_exit.place(relx=.5, rely=.94, anchor='s', height=30, width=130)

        # drawing default image
        self.canvas = Canvas(self.root, height=300, width=300)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.pack(side=TOP)

        self.root.mainloop()

    def choose_image(self):
        global path_to_image
        path_to_image = filedialog.askopenfilename(initialdir="images/", title="Select file",
                                                   filetypes=(("all files", "*.*"), ("jpeg files", "*.jpeg"),
                                                              ("png files", "*.png"), ("jpg files", "*.jpg*")))
        self.label_info.destroy()
        self.label_info = Label(master=self.root, font="Calibri 13", justify=LEFT, text=info_text_default)
        self.label_info.place(x=0, y=0)
        self.image = Image.open(path_to_image)
        self.image = self.image.resize((300, 300), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image, width=300, height=300)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.pack(side=TOP)

    def analyse(self):
        global info_text_default
        mb.showinfo("Success", "Image analysing started! Press OK and wait just a bit")
        # Here starts a classification algorithm. I used ImageAI module, that
        # can precisely enough define an object, shown on photo. Including pizzas:D
        # The model is already trained and created
        prediction = ImagePrediction()
        prediction.setModelTypeAsResNet()
        model_path = os.path.join(exec_path, "models\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
        prediction.setModelPath(model_path)
        prediction.loadModel()
        predictions, percentage_probabilities = prediction.predictImage(path_to_image, result_count=3)

        # then we modify our info text to show the results
        info_text = info_text_default + 3 * "\n"
        for index in range(len(predictions)):
            info_text += str(predictions[index]) + " : " + f"{percentage_probabilities[index]:.3f}" + '\n'
        info_text += 4 * "\n"
        if str(predictions[0]) == 'pizza':
            info_text += "Yes! This is a pizza!"
        else:
            info_text += "Nope, no pizza here:("

        # and recreate or text box with an updated information
        self.label_info.destroy()
        self.label_info = Label(master=self.root, font="Calibri 13", justify=LEFT, text=info_text)
        self.label_info.place(x=0, y=0)


app = App()
