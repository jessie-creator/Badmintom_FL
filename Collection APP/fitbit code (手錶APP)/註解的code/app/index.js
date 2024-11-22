//這是原作者的github https://github.com/gondwanasoft/fitbit-accel-fetcher
//if you want to modify and test new app remember to regenerate app_id and change name in package.json
//npx fitbit-build generate-appid
//由於原作者只有做加速度(accel),在經過我們的討論之後,我們決定加上角速度

//-------------import-------------//
///////////////下面是一樣的部分///////////////
import * as fs from "fs"
import document from 'document'
import { inbox, outbox } from 'file-transfer'

import { display } from "display"
import {goals} from "user-activity"
import { me } from "appbit"

///////////////下面是不一樣的部分///////////////
import { ACCEL_SCALAR, GYRO_SCALAR, valuesPerRecord, statusMsg, headerLength, frequency, batchPeriod } from '../common/common.js'
//把更多更常用的constant放到common中的common.js

import { Accelerometer } from "accelerometer"
import { Gyroscope } from "gyroscope";
//多import,以收集角速度

//-------------initial constant 1-------------//
///////////////下面是一樣的部分///////////////

const recordsPerBatch = frequency * batchPeriod         // 每秒100次 × 每個batch一秒
const bytesPerRecord = valuesPerRecord * 2              // 8*2 bytes/record
const recDurationPerFile = 60                           // 60 sec/file, store data into file per 60 sec
const recordsPerFile = frequency * recDurationPerFile   // 30*60 record/file
const bytesPerBatch = bytesPerRecord * recordsPerBatch  // 16*30 byte/batch

const recTimeEl = document.getElementById('recTime')
const statusEl = document.getElementById('status')
const errorEl = document.getElementById('error')
const recBtnEl = document.getElementById('recBtn')
const xferBtnEl = document.getElementById('xferBtn')
const disableTouch = true             // ignore on-screen buttons while recording (useful for swim)

///////////////下面是不一樣的部分///////////////

//這些是移到'../common/common.js'的
//const frequency = 30
//const batchPeriod = 1 

//sim是為了在模擬器中模擬手錶所以用不到
//const simSamplePeriod = 10 * Math.floor(1000 / frequency / 10)  // ms
//const isSim = goals.calories === 360  // !!

//下面是資料的儲存方式,詳細的圖可以看交接版的報告
//file:header (2 element)*(4 byte) + accelT (100 recordsPerBatch)*(2 byte) + accelX (100 recordsPerBatch)*(2 byte) + ...
//not [accelT(1),accelX(1),...,gyroZ(1)] + [accelT(2),accelX(2),...,gyroZ(2)] + ...
//it is [fileACCTstamp,fileGYROstamp] + [accelT(1),...,accelT(batchsize)] + [accelX(1),...,accelX(batchsize)] + ...
//                     + [accelT(batchsize*1 +1),...,accelT(batchsize*2)] + [accelX(batchsize*1 +1),...,accelX(batchsize*2)] + ...
const headerBuffer = new ArrayBuffer(headerLength)
const headerBufferView = new Uint32Array(headerBuffer)
var dataBuffer = new ArrayBuffer(bytesPerBatch)
var accelT = new Int16Array(dataBuffer, 0*recordsPerBatch, recordsPerBatch);
var accelX = new Int16Array(dataBuffer, 2*recordsPerBatch, recordsPerBatch);
var accelY = new Int16Array(dataBuffer, 4*recordsPerBatch, recordsPerBatch);
var accelZ = new Int16Array(dataBuffer, 6*recordsPerBatch, recordsPerBatch);
var gyroT = new Int16Array(dataBuffer, 8*recordsPerBatch , recordsPerBatch);
var gyroX = new Int16Array(dataBuffer, 10*recordsPerBatch, recordsPerBatch);
var gyroY = new Int16Array(dataBuffer, 12*recordsPerBatch, recordsPerBatch);
var gyroZ = new Int16Array(dataBuffer, 14*recordsPerBatch, recordsPerBatch);

//多import,以收集角速度
var accel = new Accelerometer({ frequency: frequency, batch: recordsPerBatch })
var gyro = new Gyroscope({ frequency: frequency, batch: recordsPerBatch })
var accelReady = false;
var gyroReady = false;

//-------------initial constant 2-------------//
//下面這邊把sim的部分去掉，另外在增加記錄gyrometer時間的常數
let fileDescriptor
let isRecording = false, isTransferring = false
let fileNumberSending
let recordsInFile, recordsRecorded
let startTime
let dateLastBatch
let fileACCTstamp   // timestamp of first record in file currently being recorded
let fileGYRTstamp
let prevACCTstamp   // timestamp of previous record, use to calculate the difference
let prevGYRTstamp
let state = {
  fileNumberRecording: undefined
}

//下面這邊都一樣
me.appTimeoutEnabled = false
restoreState()
recBtnEl.text = 'START RECORDING'
document.onkeypress = onKeyPress
recBtnEl.addEventListener("click", onRecBtn)
xferBtnEl.addEventListener("click", onXferBtn)
accel.addEventListener("reading", onAccelReading)
gyro.addEventListener("reading", onGyroReading)
inbox.addEventListener("newfile", receiveFilesFromCompanion)
receiveFilesFromCompanion()
if (state.fileNumberRecording && fs.existsSync('1')) {
  xferBtnEl.text = 'TRANSFER TO PHONE'
  xferBtnEl.style.display = 'inline'
}

//************* User input ***************
//這邊是決定在不同的情況下(收資料or傳資料),要怎麽處理User input
//onRecBtn:手錶上中間,藍色的按鈕
//onXferBtn:取消傳輸的按鈕
//onKeyPress:手錶左邊,按了會振動的按鈕

function onRecBtn() {
  if (isTransferring) return
  if (disableTouch && isRecording) return

  if (isRecording) stopRec()
  else startRec()
}

function onXferBtn() {
  if (isRecording) return

  if (isTransferring) stopTransfer()
  else startTransfer()
}

function onKeyPress(e) {
  //console.log('onKeyPress');
  if (isRecording) {
    stopRec()
    e.preventDefault()
  }
}

//************* Record data ***************
//因為simAccelTick()沒有用,所以我把它刪了

//create new file, reserve 2*(Uint32) to save initial timestamp (fileACCTstamp & fileGYRTstamp)
function openFile(){
  console.log(`Starting new file: ${state.fileNumberRecording}`)
  fileDescriptor = fs.openSync(state.fileNumberRecording, 'a')
  headerBufferView[0] = fileACCTstamp
  headerBufferView[1] = fileGYRTstamp
  fs.writeSync(fileDescriptor, headerBuffer)
  recordsInFile = 0
  statusEl.text = 'Recording file '+ state.fileNumberRecording
  display.poke()
}


//加速度的讀取函式,因為前面有用accel.addEventListener("reading", onAccelReading)
//因此當在startRec()中,進行accel.start()時,就會開始記錄
//accel object reading function
function onAccelReading(){
  if(!isRecording){
    console.error("onAccelReading but not recording")
    return
  }

  const dateNow = Date.now()
  if(dateLastBatch){
    //console.log(`t since last batch: ${dateNow-dateLastBatch} ms`)  // debugging
  }
  dateLastBatch = dateNow

  const needNewFile = fileDescriptor === undefined || recordsInFile >= recordsPerFile
  if(needNewFile){
    fileACCTstamp = prevACCTstamp = accel.readings.timestamp[0]
    console.log(`needNewFile: fileACCTstamp=${fileACCTstamp}`);
  }

  //這個程式傳時間的方式其實是先將開始的時間(fileTimestamp)記入headerBufferView
  //之後每一個record所傳遞的時間資料其實是與前一個record的時間差
  //因此prevACCTstamp原先是用來記錄前一個record的時間
  //但是因爲時間序列有bug,因此我就想說直接手動寫入時間
  //frequency=100,因此一秒會被切成10毫秒(accelT[index] = 10)
  //下面注解掉的程式便是用來傳輸時間的
  //然後原作者的程式碼有一些debug的code,可以參考一下
  //不過老師之前有說這個bug的影響不大
  
  //另外,因為databufferview是傳Int16
  //因此會先把加速度(like 0.5m/s2 )乘上ACCEL_SCALAR(500),轉成整數(like 250)
  //等到在寫入txt檔案時再除回去

  let acctimestamp
  for(let index = 0; index < recordsPerBatch; index++){
    // acctimestamp = accel.readings.timestamp[index]
    // accelT[index] = acctimestamp - prevACCTstamp      //this is correct, save interval length
    // prevACCTstamp = acctimestamp
    accelT[index] = 10

    accelX[index] = accel.readings.x[index] * ACCEL_SCALAR    // data is multiple by scalar because we need to transfer interger
    accelY[index] = accel.readings.y[index] * ACCEL_SCALAR
    accelZ[index] = accel.readings.z[index] * ACCEL_SCALAR
  }

  //這邊的部分,因為我們除了accel之外還有gyro,為了避免狀態有錯
  //我是等到兩者都進入ready之後在統一開始進行寫入
  //那因為我們有多一倍的資料因此寫入的資料量是recordsPerBatch*bytesPerRecord
  //跟原作者的batchSize*bytesPerRecord不同,請注意
  //write to file/open neww file when two object are all ready
  accelReady = true;
  if(accelReady && gyroReady){
    accelReady = false;
    gyroReady = false;

    if(fileDescriptor === undefined){
      openFile()
    } else {
      if (recordsInFile >= recordsPerFile) {
        fs.closeSync(fileDescriptor)
        recordsRecorded += recordsInFile
        state.fileNumberRecording++
        openFile()
      }
    }

    try{
      fs.writeSync(fileDescriptor, dataBuffer, 0, recordsPerBatch*bytesPerRecord)
      recordsInFile += recordsPerBatch
    }catch(e){
      console.error("Can't write to file (out of storage space?)")
    }

    recTimeEl.text = Math.round((Date.now()-startTime)/1000)
  }
}

//角速度的版本,跟加速度的大同小異
//gyro object reading function
function onGyroReading(){
  if(!isRecording){
    console.error("onGyroReading but not recording")
    return
  }

  const dateNow = Date.now()
  if(dateLastBatch){
    //console.log(`t since last batch: ${dateNow-dateLastBatch} ms`)  // debugging
  }
  dateLastBatch = dateNow

  const needNewFile = fileDescriptor === undefined || recordsInFile >= recordsPerFile
  if(needNewFile){
    fileGYRTstamp = prevGYRTstamp = gyro.readings.timestamp[0]
    console.log(`needNewFile: fileGYRTstamp=${fileGYRTstamp}`);
  }

  let gyrtimestamp
  for(let index = 0; index < recordsPerBatch; index++){
    // gyrtimestamp = gyro.readings.timestamp[index]
    // gyroT[index] = gyrtimestamp - prevGYRTstamp
    // prevGYRTstamp = gyrtimestamp

    gyroT[index] = 10

    gyroX[index] = gyro.readings.x[index] * GYRO_SCALAR
    gyroY[index] = gyro.readings.y[index] * GYRO_SCALAR
    gyroZ[index] = gyro.readings.z[index] * GYRO_SCALAR
  }

  gyroReady = true;
  if(accelReady && gyroReady){
    accelReady = false;
    gyroReady = false;

    if(fileDescriptor === undefined){
      openFile()
    } else {
      if (recordsInFile >= recordsPerFile) {
        fs.closeSync(fileDescriptor)
        recordsRecorded += recordsInFile
        state.fileNumberRecording++
        openFile()
      }
    }

    try{
      fs.writeSync(fileDescriptor, dataBuffer, 0, recordsPerBatch*bytesPerRecord)
      recordsInFile += recordsPerBatch
    }catch(e){
      console.error("Can't write to file (out of storage space?)")
    }

    recTimeEl.text = Math.round((Date.now()-startTime)/1000)
  }
}

//下面基本上就是在startRec時觸發accel和gyro,然後再stopRec時把他們停掉
//另外,我之前在debug時有在deletefile的地方加一些歸零的東西
function startRec(){
  if(isTransferring) return

  deleteFiles()

  dateLastBatch = recordsInFile = recordsRecorded = 0
  recTimeEl.text = '0'
  state.fileNumberRecording = 1
  errorEl.style.fill = '#ff0000'
  errorEl.text = ''
  statusEl.text = 'Recording file ' + state.fileNumberRecording
  accel.start()   //trigger accel object reading function
  gyro.start()
  console.log('Started.')
  recBtnEl.text = disableTouch? '← PRESS KEY TO STOP' : 'STOP RECORDING'
  recBtnEl.state = 'disabled'
  recBtnEl.style.display = 'inline'
  xferBtnEl.style.display = 'none'
  startTime = Date.now()
  isRecording = true
}

function deleteFiles(){
  accelT.fill(0);
  accelX.fill(0);
  accelY.fill(0);
  accelZ.fill(0);
  gyroT.fill(0);
  gyroX.fill(0);
  gyroY.fill(0);
  gyroZ.fill(0);

  const fileIter = fs.listDirSync('/private/data')
  let nextFile = fileIter.next()
  while(!nextFile.done){
    fs.unlinkSync(nextFile.value)
    nextFile = fileIter.next()
  }
}

function stopRec(){
  accel.stop()    //remember to stop the reading file
  gyro.stop()

  fs.closeSync(fileDescriptor)
  fileDescriptor = undefined

  console.log(`stopRec(): fileNumberRecording=${state.fileNumberRecording} recordsInFile=${recordsInFile}`)
  if(!recordsInFile){
    console.error(`Empty file!`)
    fs.unlinkSync(state.fileNumberRecording)
    state.fileNumberRecording--
  }
  recordsRecorded += recordsInFile
  console.log('Stopped.')
  statusEl.text = `Recorded ${state.fileNumberRecording} file(s)`
  const size = recordsRecorded * bytesPerRecord / 1024
  errorEl.style.fill = '#0080ff'
  errorEl.text = `(${recordsRecorded} readings; ${Math.round(size)} kB)`
  display.poke()
  recBtnEl.text = 'START RECORDING'
  recBtnEl.style.display = 'inline'
  recBtnEl.state = 'enabled'
  if (state.fileNumberRecording) {
    xferBtnEl.text = 'TRANSFER TO PHONE'
    xferBtnEl.style.display = 'inline'
  }
  isRecording = false
}

//下面是傳輸資料的code,我基本沒動過
//*************Transfer data ***************

function startTransfer() {
  if (!state.fileNumberRecording) return

  isTransferring = true
  errorEl.style.fill = '#ff0000'
  errorEl.text = ''
  recTimeEl.text = ''
  recBtnEl.text = ''
  recBtnEl.style.display = 'none'
  xferBtnEl.text = 'ABORT TRANSFER'
  xferBtnEl.style.display = 'inline'
  fileNumberSending = 1
  sendFile()
}

function stopTransfer() {
  statusEl.text = 'Transfer aborted'
  display.poke()
  errorEl.text = ''
  recBtnEl.text = 'START RECORDING'
  recBtnEl.style.display = 'inline'
  xferBtnEl.text = 'TRANSFER TO PHONE'
  xferBtnEl.style.display = 'inline'
  isTransferring = false
}

function sendFile(fileName) {
  // Sends  fileName (if specified) or fileNumberSending
  // File transfer is more reliable than messaging, but has higher overheads.
  // If you want to send data very frequently and/or with less latency,
  // use messaging (and accept the risk of non-delivery).
  // TODO 3.5: If companion doesn't get launched, use timeout to report failure. What is last log line before hanging? What is next log line if not hanging?
  // TODO 3.6: If companion doesn't respond, transfer to a relanching app, restart and resume

  const operation = fileName? 'Res' : 'S'   // plus 'ending...'
  if (!fileName) fileName = fileNumberSending

  outbox
    .enqueueFile("/private/data/"+fileName)
    .then(ft => {
      statusEl.text = operation + 'ending file ' + fileName + ' of ' + state.fileNumberRecording + '...'
      display.poke()
      console.log(`${operation}ending file ${fileName} of ${state.fileNumberRecording}: queued`);
    })
    .catch(err => {
      console.error(`Failed to queue transfer of ${fileName}: ${err}`);
      errorEl.text = "Can't send " + fileName + " to companion"
      display.poke()
    })
}

function sendObject(obj) {
  // File transfer is more reliable than messaging, but has higher overheads.
  // If you want to send data very frequently and/or with less latency,
  // use messaging (and accept the risk of non-delivery).
  fs.writeFileSync("obj.cbor", obj, "cbor")

  outbox
    .enqueueFile("/private/data/obj.cbor")
    .then(ft => {
      console.log(`obj.cbor transfer queued.`);
    })
    .catch(err => {
      console.log(`Failed to schedule transfer of obj.cbor: ${err}`);
      errorEl.text = "Can't send status to companion"
      display.poke()
    })
}

function sendData(data) {
  // File transfer is more reliable than messaging, but has higher overheads.
  // If you want to send data very frequently and/or with less latency,
  // use messaging (and accept the risk of non-delivery).

  fs.writeFileSync("data.txt", data, "utf-8")

  outbox
    .enqueueFile("/private/data/data.txt")
    .then(ft => {
      //console.log(`Transfer queued.`);
    })
    .catch(err => {
      //console.log(`Failed to schedule transfer: ${err}`);
    })
}

function receiveFilesFromCompanion() {
  let fileName
  while (fileName = inbox.nextFile()) {
    console.log(`receiveFilesFromCompanion(): received ${fileName}`)
    const response = fs.readFileSync(fileName, 'cbor')
    console.log(`watch received response status code ${response.status} (${statusMsg[response.status]}) for file ${response.fileName}`)
    // See /common/common.js for response.status codes.
    if (response.fileName) {
      if (isTransferring) {
        if (response.status === 200) sendNextFile()
        else resendFile(response)
      }
    } else {  // no fileName; must have been a control object
      // should check response.status
      statusEl.text = 'Finished — see phone'
      display.poke()
      recBtnEl.text = 'START RECORDING'
      recBtnEl.style.display = 'inline'
      xferBtnEl.style.display = 'none' // xferBtnEl.text = ''
      isTransferring = false
    }

    fs.unlinkSync(fileName)
  }
}

function sendNextFile() {
  errorEl.text = ''
  if (++fileNumberSending > state.fileNumberRecording) {
    console.log('All files sent okay; waiting for server to acknowledge')
    statusEl.text = 'All data sent; wait...'
    display.poke()
    sendObject({status:'done'})
    return
  }

  sendFile()
}

function resendFile(response) {
  errorEl.text = `${statusMsg[response.status]} on ${response.fileName}`
  display.poke()
  console.log(`Resending ${response.fileName}`)
  sendFile(response.fileName)
}

me.onunload = () => {
  saveState()
}

function saveState() {
  fs.writeFileSync("state.cbor", state, "cbor")
}

function restoreState() {
  // Returns true if state restored.
  let newState;
  try {
    newState = fs.readFileSync("state.cbor", "cbor");
    state = newState;
    return true
  } catch(err) {   // leave state as is
  }
}
// TODO 3.9 android-fitbit-fetcher needs a way to reset; currently can receive files from different sessions.