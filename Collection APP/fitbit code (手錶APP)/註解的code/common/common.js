//因為Int16的bit數只有十六個(廢話)
//可以透過ACCEL_SCALAR來控制資料的解析度
//ACCEL_SCALAR越大,記錄的數字就越大,只是小數點的部分會越小
export const ACCEL_SCALAR = 500   // up to 6.5g; resolution 0.002 m/s/s
export const GYRO_SCALAR = 500
export const valuesPerRecord = 8  // (x, y, z, time)*2
export const frequency = 100      // record/sec
export const batchPeriod = 1      // sec/batch 一秒一batch
export const statusMsg = {        // codes<100 are only used from companion to watch; codes>550 are custom HTTP codes sent from android-fitbit-fetcher
  1:"Server didn't respond",
  2:"Server comm error",
  3:"Server comm reject",
  4:"Server response bad",
  200:"OK",
  500:'Server error',
  501:'Not implemented',
  555:'Invalid data',
  556:'Invalid length'
}
export const headerLength = 2 * Uint32Array.BYTES_PER_ELEMENT // 4, one Unit32 for fileTimestamp