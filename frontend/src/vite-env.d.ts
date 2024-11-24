/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_ENABLE_API_KEYS: string
  readonly VITE_ENABLE_2FA: string
  readonly VITE_ENABLE_NOTIFICATIONS: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
