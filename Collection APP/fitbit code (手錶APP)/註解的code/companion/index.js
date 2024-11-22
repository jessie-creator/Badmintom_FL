import { encode } from 'cbor'
import { me as companion } from "companion"
import { inbox, outbox } from "file-transfer"
import { localStorage } from "local-storage"
import { settingsStorage } from "settings"
import { ACCEL_SCALAR, GYRO_SCALAR, valuesPerRecord, statusMsg, headerLength, frequency, batchPeriod } from '../common/common.js'

//headerbufferlength:如同報告提到的是4
const httpURL = 'http://127.0.0.1:3000'
const headerBufferLength = headerLength / 2   // 8/2, buffer is 16-bit array
const recordsPerBatch = frequency * batchPeriod

let responseTimeoutTimer
let fileNbrPrev
let acctimestamp = 0
let gyrtimestamp = 0

async function receiveFilesFromWatch() {
  console.log('receiveFilesFromWatch()')
  let file
  while ((file = await inbox.pop())) {
    console.log(`Received file ${file.name}`)

    if (file.name === 'obj.cbor') receiveStatusFromWatch(file)
    else receiveDataFromWatch(file)
  }
}

async function receiveDataFromWatch(file){
  if (file.name === '1'){
    fileNbrPrev = 0
    acctimestamp = 0
    gyrtimestamp = 0
  }

  var data = await file.arrayBuffer()

  var headerBufferView = new Uint32Array(data)
  // get the start timestamp to add up
  // let acctimestamp = headerBufferView[0]
  // let gyrtimestamp = headerBufferView[1]

  var dataBufferView = new Int16Array(data)
  //(total_length - header_length)/per_record_length
  const recordCount = (dataBufferView.length - headerBufferLength) / valuesPerRecord

  console.log(`Got file ${file.name}; contents: ${data.byteLength} bytes = ${dataBufferView.length} elements = ${recordCount} accel records;  timestamp = ${acctimestamp}`)
  settingsStorage.setItem('fileNbr', file.name)

  const fileNbr = Number(file.name)
  if (fileNbr !== fileNbrPrev + 1) console.log(`File received out of sequence: prev was ${fileNbrPrev}; got ${fileNbr}`)
  fileNbrPrev = fileNbr

  let elementIndex = headerBufferLength   //we need to eliminate fileACCTstamp & fileGYRTstamp
  let record
  let content = ''
  let acctstampDiff = 0
  let gyrtstampDiff = 0
  let batchcount = 0
  //access stored daata according to previous saving format
  for(let recordIndex = 0; recordIndex < recordCount; ){
    for(let index=0; index < recordsPerBatch; index++ , recordIndex++){
      
      //如同在app/index中提到的我是先把時間的間隔設成10
      //詳細的資料儲存方式請看報告
      
      // acctstampDiff = dataBufferView[elementIndex + (batchcount*8 + 0)*recordsPerBatch + index]
      acctstampDiff = 10
      acctimestamp += acctstampDiff
      // gyrtstampDiff = dataBufferView[elementIndex + (batchcount*8 + 4)*recordsPerBatch + index]
      gyrtstampDiff = 10
      gyrtimestamp += gyrtstampDiff

      record = `${acctimestamp},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 1)*recordsPerBatch + index]/ACCEL_SCALAR},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 2)*recordsPerBatch + index]/ACCEL_SCALAR},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 3)*recordsPerBatch + index]/ACCEL_SCALAR},`
      content += record
      record = `        ${gyrtimestamp},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 5)*recordsPerBatch + index]/GYRO_SCALAR},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 6)*recordsPerBatch + index]/GYRO_SCALAR},`
      content += record
      record = `${dataBufferView[elementIndex + (batchcount*8 + 7)*recordsPerBatch + index]/GYRO_SCALAR}\r`
      content += record
    }
    batchcount ++
  }

  sendToServer(content, file.name)

  localStorage.setItem('fileNbrPrev', fileNbrPrev)
}

//下面的部分我沒有動到

async function receiveStatusFromWatch(file) {
  const status = await file.cbor()
  console.log(`status=${status} (${typeof status})`)
  const statusText = status.status
  console.log(`receiveStatusFromWatch() status=${statusText}`)
  settingsStorage.setItem('fileNbr', `Watch: ${statusText}`)
  sendToServer(JSON.stringify(status), null, true)
}

;(function() {
  companion.wakeInterval = 300000   // encourage companion to wake every 5 minutes

  // Extract persistent global variables from localStorage:
  fileNbrPrev = localStorage.getItem('fileNbrPrev')
  if (fileNbrPrev == null) fileNbrPrev = 0; else fileNbrPrev = Number(fileNbrPrev)

  inbox.addEventListener("newfile", receiveFilesFromWatch)
  receiveFilesFromWatch()
})()

function sendToServer(data, fileName, asJSON) {
  // fileName can be null if sending a status message.
  console.log(`sendToServer() fileName=${fileName} asJSON=${asJSON}`)
  const headers = {}
  if (fileName) headers.FileName = fileName
  if (asJSON) headers["Content-Type"] = "application/json"
  //let fetchInit = {method:'POST', headers:{"FileName":fileName}, body:data}
  let fetchInit = {method:'POST', headers:headers, body:data}
  // To send binary data, use {method:'POST', headers:{"Content-type": "application/octet-stream"}, body:data}

  // timeout in case of no exception or timely response
  responseTimeoutTimer = setTimeout(() => {
    responseTimeoutTimer = undefined
    console.log(`onResponseTimeout()`)
    sendToWatch(fileName, 1, true)   // server response timeout
  }, 5000);

  fetch(httpURL, fetchInit)
    .then(function(response) {    // promise fulfilled (although server response may not be Ok)
      console.log(`sendToServer() fetch fulfilled: fileName=${fileName}; ok=${response.ok}; status=${response.status}; sText=${response.statusText}`)
      if (responseTimeoutTimer !== undefined) {clearTimeout(responseTimeoutTimer); responseTimeoutTimer = undefined}
      sendToWatch(fileName, response.status)
      if (response.ok) {
        serverResponseOk(fileName, response.statusText)
      } else {
        serverResponseError(response.status, response.statusText)
      }
    }, function(reason) {       // promise rejected (server didn't receive file correctly, or no server because running in simulator)
      console.error(`sendToServer() fetch rejected: ${reason}; fileName=${fileName}. Ensure server is running.`)
      if (responseTimeoutTimer !== undefined) {clearTimeout(responseTimeoutTimer); responseTimeoutTimer = undefined}
      sendToWatch(fileName, 3, true)    // TODO 8 should be 3; set to 200 to allow device and companion testing in sim (ie, without android)
    })
    .catch(function(err) {    // usually because server isn't running
      console.error(`sendToServer() fetch catch: fileName=${fileName}; error: ${err}. Ensure server is running.`)
      if (responseTimeoutTimer) {clearTimeout(responseTimeoutTimer); responseTimeoutTimer = undefined}
      sendToWatch(fileName, 2, true)
    })

  console.log(`sendToServer() sent ${fileName}`)
}

function serverResponseOk(fileName, text) {
  console.log(`serverResponseOk(): text=${text}`)
  const statusText = fileName? 'OK' : 'Server: done'
  settingsStorage.setItem('status', statusText)
}

function serverResponseError(status, text) {
 console.error(`serverResponseError(): status=${status} text=${text}`)
 settingsStorage.setItem('status', statusMsg[status])
}

function sendToWatch(fileName, status, updateSettings) {
  if (updateSettings) settingsStorage.setItem('status', statusMsg[status])

  outbox.enqueue('response-'+Date.now(), encode({fileName:fileName, status:status}))
  .then((ft) => {
    console.log(`Transfer of ${ft.name} successfully queued.`);
  })
  .catch((error) => {
    console.error(`Failed to queue response for ${fileName}: ${error}`);
    settingsStorage.setItem('status', "Can't send to watch")
  })
}