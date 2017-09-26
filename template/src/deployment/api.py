from flask import request, jsonify
from flask.ext.api import FlaskAPI
from flask_cors import CORS
from ..main import load

app = FlaskAPI(__name__)
cors = CORS(app)

execute_func = load.initialize_func()


@app.route("/demo", methods=['POST'])
def get_demo():
  """
  Simple synchronous endpoint for demo.

  Make query to 0.0.0.0:5000/query with
  POST protocol, payload: {"query": "hi"}.
  Response: {"response": "hello"}
  """
  text = str(request.data.get('query', ''))
  response = execute_func(text)
  return jsonify({'response': response})


if __name__ == "__main__":
  app.run(host='0.0.0.0')

