import bottle
from model import Model, Params


def answering_function(passage, question):
    # TODO
    return passage


app = bottle.Bottle()


@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['passage']
    return {"answer": answering_function(passage, question)}


@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html


if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
