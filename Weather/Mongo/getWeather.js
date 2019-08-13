// Script to retrieve weather data from Dark Sky and put it into the database

const Mongo = require('mongodb')
const request = require('request')
const moment = require('moment')
const csv = require('csvtojson')


let routeFilename = `${__dirname}/routeInterpolated.csv`

let apiKey = 'b751290bc7f81afa715bdb6d5eac71ed'
let requestPrefix = 'https://api.darksky.net/forecast'

// Retrieve
var MongoClient = Mongo.MongoClient;

// Connect to the db
MongoClient.connect("mongodb://localhost:27017/", { useNewUrlParser: true }, async function(err, db) {
  if(err) {
    console.log("Error connecting to MongoDB");
    process.exit(-1)
  } 
  else console.log(`Connected to MongoDB`)
  DB_weather = db.db('CUER').collection(`weather`)


  let route = await csv().fromFile(routeFilename)

  let remainingLocations = route
  let fetchingLocations = []
  let fetchingPromiseList = []
  let processingDistances = []
 
  completed = {}
  for (let location of route) {
    location.distance = parseFloat(location.distance)

    completed[`${location.distance}`] = {
      daily: 0,
      hourly: 0,
      fullDone: false,
      dailyDone: false,
      hourlyDone: false,
      currentlyDone: false
    }
  }

  // Retrieve the data for each location specified
  while (remainingLocations.length > 0) {

    let location = remainingLocations.shift()

    // User defined custom skip
    if (location.distance < 0) continue

    let promise = new Promise(function(resolve, reject) {
      let requestString = `${requestPrefix}/${apiKey}/${location.latitude},${location.longitude}?units=si&extend=hourly`

      request(requestString, { json: true }, function(err, response, body) {
        if (response.statusCode == 200) resolve(body)
        else reject()
      })
      


    })

    fetchingLocations.push(location)
    fetchingPromiseList.push(promise)

    if (fetchingPromiseList.length == 1 || remainingLocations.length == 0) {

      let promiseResponses = await promiseAll(fetchingPromiseList)

      let promiseMongoUpdates = []

      for (let responseIndex in promiseResponses) {
        let response = promiseResponses[responseIndex]
        let thisLocation = fetchingLocations[responseIndex]
        
        // For any requests that failed, put them back on the list
        if (response == undefined) {
          remainingLocations.push(thisLocation)
        }

        // If the response was good, add it to the database
        else {

          console.log(`Processing dist = ${thisLocation.distance}, lat = ${thisLocation.latitude}, long = ${thisLocation.longitude}`)
          processingDistances.push(thisLocation.distance)

          let currentTime = moment()
          let additionalParameters = {
            _updated: currentTime.unix(),
            _docType: `full`,
            _distance: thisLocation.distance,
            _timeString: currentTime.format(),
            _updatedString: currentTime.format(),
          }

          let doc = Object.assign(response,  additionalParameters)

          promiseMongoUpdates.push(mongo_updateOne(DB_weather, 
            {
              _docType: doc._docType,
              latitude: doc.latitude,
              longitude: doc.longitude,
              currently: {time: doc.currently.time}
            },
            doc, 
            { upsert: true }, 
            function (err, result) {
              if (err) throw err;
              console.log(`Dist = ${doc._distance}, Inserted full document`);
              completed[`${doc._distance}`].fullDone = true
          }))

          let quantityData = {
            hourly: response.hourly.data.length,
            daily: response.daily.data.length
          }

          for (let docType of ['hourly', 'daily']) {
            for (let data of response[docType].data) {

              let additionalParameters = {
                _updated: currentTime.unix(),
                _docType: docType,
                _distance: thisLocation.distance,
                _timeString: moment.unix(data.time).format(),
                _updatedString: currentTime.format(),
                latitude: response.latitude,
                longitude: response.longitude,
                timezone: response.timezone,
                offset: response.offset,
                ...response.flags,
              }
              let doc = Object.assign(data, additionalParameters)

              promiseMongoUpdates.push(mongo_updateOne(DB_weather, 
                {
                  _docType: doc._docType,
                  latitude: doc.latitude,
                  longitude: doc.longitude,
                  time: doc.time,
                }, 
                doc, 
                { upsert: true }, 
                function (err, result) {
                if (err) throw err;
                completed[`${doc._distance}`][docType] += 1
                // console.log(`${completed[`${doc._distance}`][docType]}/${quantityData[docType]}`)
                if (completed[`${doc._distance}`][docType] == quantityData[docType]) {
                  console.log(`Dist = ${doc._distance}, Inserted all ${docType} documents`);
                  completed[`${doc._distance}`][`${docType}Done`] = true
                }
              }))
            }
          }

          // Also save the "currently" data as if it is hourly data
          let additionalParametersCurrently = {
            _updated: currentTime.unix(),
            _docType: `hourly`,
            _distance: thisLocation.distance,
            _timeString: currentTime.format(),
            _updatedString: currentTime.format(),
            latitude: response.latitude,
            longitude: response.longitude,
            timezone: response.timezone,
            offset: response.offset,
            ...response.flags,
          }
          let docCurrently = Object.assign(response.currently, additionalParametersCurrently)

          promiseMongoUpdates.push(mongo_updateOne(DB_weather, 
            {
              _docType: docCurrently._docType,
              latitude: docCurrently.latitude,
              longitude: docCurrently.longitude,
              time: docCurrently.time,
            },
            docCurrently,
            { upsert: true },
            function (err, result) {
              if (err) throw err;
              console.log(`Dist = ${docCurrently._distance}, Inserted current document`);
              completed[`${docCurrently._distance}`].currentlyDone = true
            }))

        }
      }

      await promiseAll(promiseMongoUpdates)
      
      fetchingLocations = []
      fetchingPromiseList = []
      processingDistances = []
      promiseMongoUpdates = []

    }
    

  }
  


});

// Same as Promise.all but always returns an array of results even if some of them rejected
// Resolved promises fulfill to their resolved value
// Rejected promises fulfull to undefined
// Taken from https://davidwalsh.name/promises-results
function promiseAll(arr) {
  return Promise.all(arr.map( p => p.catch(() => undefined)))
}

async function mongo_updateOne(collection, filter, doc, options, callback) {
  return new Promise(function(resolve, reject) {
    collection.updateOne(
    filter,
    { $set: doc },
    options,
    function(err, result) {
      callback(err, result)
      if (err) reject(err)
      resolve()
    }
  )
  })
  
}





