** REST API FACE RECOGNITION **

This service is supposed to use by MyAttendance Mobile App

How to run this project : nodemon index.js / node index.js
then open -> http://localhost:5000 (or any port you want)

There's some endpoint need to know :
- "/" -> root endpoint
- "/recognizing-face" -> Registering face to Model and Database
- "/recognizer-face" -> Checking if a face in image is registered in model and database
- "/search-face/:label" -> Checking the registered face data by given label/name in collection of database
- "/delete-face/:label" -> Delete the registered face data by given label/namem in collection of database

Regards : Bagus Subagja (11/10/2023)