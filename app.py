from flask import Flask, render_template, request
from calculator.calculator import solve_limit, plot_limit
import os
import shutil

from flask import Flask, render_template, request, jsonify  # Importe o jsonify

app = Flask(__name__)

plot_index = 0
images = []
results = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global plot_index, results, images, error

    diretorio = 'static/images/grafics/'

    if os.path.exists(diretorio):
        for filename in os.listdir(diretorio):
            file_path = os.path.join(diretorio, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                error = print(f'Falha ao deletar {file_path}. Motivo: {e}')
    else:
        error = (f'O diretório {diretorio} não existe.')

    if request.method == 'POST':
        user_input = request.form['input']

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                solved = solve_limit(user_input)
                if isinstance(solved, tuple):
                    result, expr, sym, point = solved
                else:
                    error = solved
                    return jsonify(error=error)
                plot_limit(function=expr, sym=sym, index=plot_index, point=point)

                results.clear()
                images.clear()
                
                results.append(result)

                images = [f'static/images/grafics/{plot_index}-plot.png']

                plot_index += 1

                return jsonify(results=results, images=images)

            except Exception as e:
                error = f'Error: {str(e)}'
                print(error)

                return jsonify(results=results, images=images, error=error)

        else:
            return render_template("index.html")

    else:
        return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)