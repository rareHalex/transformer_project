from flask import Flask, redirect, request, session
from flask import render_template
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter
import predict_model
from extra_models import inference_models

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
metrics = PrometheusMetrics(app)
PREDICTION_COUNT = Counter("predictions_total", "Number of predictions", ["label"])


@app.route('/', methods=['POST', 'GET'])
@app.route('/main', methods=['POST', 'GET'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("api_invocations_total", "number of invocations")
def main():
    if request.method == 'POST':
        title = request.form['title']
        abstract = request.form['abstract']
        session['title'] = title
        session['abstract'] = abstract
        return redirect('/predict')
    else:
        return render_template('main.html')


@app.route('/predict')
def predict():
    title = str(session.get('title', None))
    abstract = str(session.get('abstract', None))
    prediction_array = predict_model.prediction_model(title, abstract)
    for targets in prediction_array:
        PREDICTION_COUNT.labels(label=targets).inc()
    return render_template('predict.html', title=title, text=abstract, pred_arr=prediction_array)


@app.route('/generation')
def generation():
    abstract = str(session.get('abstract', None))
    new_title = inference_models.title_generation(abstract)
    return render_template('generation.html', title=new_title)


@app.route('/about')
def about():
    return render_template('about_fl.html')


if __name__ == '__main__':
    app.run()


