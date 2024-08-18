from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from secret_key import key
import os, shutil, test

app = Flask(__name__)
app.config["SECRET_KEY"] = key
app.config["UPLOAD_FOLDER"] = "test_song"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    result = None
    if form.validate_on_submit():
        file = form.file.data
        folder = "test_song"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["UPLOAD_FOLDER"],
                secure_filename(file.filename),
            )
        )
        # run the neural network here
        result = test.run()
    return render_template("index.html", form=form, result=result)


if __name__ == "__main__":
    app.run(debug=True)
