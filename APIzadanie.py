from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import numpy as np
import urllib.request

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#liczenie z linku
class CounterLink(Resource):
    def get(self):
        image_url = request.args.get('image_url')

        if not image_url:
            return{'error:' 'Paramenter image_url musi zostać podany'},400

        try:
            resp = urllib.request.urlopen(image_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))
            return {'count': len(boxes)}

        except Exception as e:
            return {'error': f'Wystąpił błąd: {str(e)}'}, 500



class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('dworzec.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))

        return {'count': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class PeopleCounter2(Resource):
    def post(self):
        try:
            data = request.get_data()
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {'error': 'Nie udało się wczytać obrazu'}, 500

            boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))

            return {'count': len(boxes)}
        except Exception as e:
            return {'error': f'Wystąpił błąd: {str(e)}'}, 500

api.add_resource(PeopleCounter2, '/people-counter')
#w terminalu curl -X POST -H "Content-Type: image/jpeg" --data-binary "@/Users/izamatejko/Desktop/dworzec.jpg" http://127.0.0.1:5001/people-counter
api.add_resource(CounterLink, '/B')
#http://127.0.0.1:5001/B?image_url=https://www.wrotapodlasia.pl/resource/image/2/300/156565/1371536/828x0.jpg
api.add_resource(PeopleCounter, '/A')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True, port=5001 )
