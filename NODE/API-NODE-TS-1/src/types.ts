// export type Weather = 'sunny' | 'rainy' | 'cloudy' | 'windy' | 'stormy'

// export type Visibility = 'great' | 'good' | 'ok' | 'poor'

export enum Weather {
  Sunny = 'sunny',
  Rainy = 'rainy',
  Cloudy = 'cloudy',
  Windy = 'windy',
  Stormy = 'stormy'
}

export enum Visibility {
  Great = 'great',
  Good = 'good',
  Ok = 'ok',
  Poor = 'poor'
}

export interface diaryEntry {
  id: number
  date: string
  weather: Weather
  visibility: Visibility
  comment: string
}

// crear un tipo basado en la interfaz diaryEntry pero sin el campo comment
export type NonSensitiveInfoDiaryEntry = Omit<diaryEntry, 'comment'>

export type NewDiaryEntry = Omit<diaryEntry, 'id'> // crear un tipo basado en la interfaz diaryEntry pero sin el campo id
