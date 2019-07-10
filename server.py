"""

# @author: Yue Peng
# @email: yuepaang@gmail.com
# @createTime: 2018-11-06, 17:39:17 GMT+0800
# @description: Flask API for Testing

# Browser 
http://54.227.110.43:5000/predict?msg=HelloWorld

# Curl
>curl -X POST -H "Content-Type: application/json" -d "{ \"msg\":
\"Hello World\" }" http://54.227.110.43:5000/predict

# Response 
{
  "response": "Hello World",
  "success": true
}
"""
import flask
app = flask.Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():
	data = {"success": False}

	params = flask.request.json
	if (params is None):
		params = flask.request.args

	if (params is not None):
		data["response"] = params.get("msg")
		data["success"] = True

	return flask.jsonify(data)


app.run(host='0.0.0.0', port=5000)
