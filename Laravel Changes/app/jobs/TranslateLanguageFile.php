<?php

namespace App\Jobs;

use Illuminate\Bus\Batchable;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;
use Symfony\Polyfill\Intl\Idn\Info;

class TranslateLanguageFile implements ShouldQueue
{
    use Batchable, Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    protected $fileName;
    protected $primaryLang;
    protected $secondaryLang;

    /**
     * The number of times the job may be attempted.
     *
     * @var int
     */
    public $tries = 3;

    /**
     * The number of seconds to wait before retrying the job.
     *
     * @var int
     */
    public $backoff = 30;

    /**
     * The maximum number of seconds the job can run before timing out.
     *
     * @var int
     */
    public $timeout = 300; // 5 minutes

    /**
     * Create a new job instance.
     */
    public function __construct(string $fileName, string $primaryLang, string $secondaryLang)
    {
        $this->fileName = $fileName;
        $this->primaryLang = $primaryLang;
        $this->secondaryLang = $secondaryLang;
    }

    /**
     * Execute the job.
     */
    public function handle(): void
    {
        $logContext = [
            'file' => $this->fileName,
            'primary_lang' => $this->primaryLang,
            'secondary_lang' => $this->secondaryLang
        ];

        // Check if batch was cancelled
        if ($this->batch() && $this->batch()->cancelled()) {
            Log::warning("Translation job cancelled for file", array_merge($logContext, ['reason' => 'batch_cancelled']));
            return;
        }

        try {
            $primaryPath = base_path("resources/lang/{$this->primaryLang}/{$this->fileName}.php");
            $secondaryPath = base_path("resources/lang/{$this->secondaryLang}/{$this->fileName}.php");

            // Update current file in progress
            $progress = Cache::get('translation_import_progress', []);
            $progress['current_file'] = $this->fileName;
            Cache::put('translation_import_progress', $progress, now()->addHours(2));

            // Load primary language file
            if (!File::exists($primaryPath)) {
                Log::error("Translation job failed - primary language file not found", array_merge($logContext, [
                    'primary_path' => $primaryPath,
                    'status' => 'failed',
                    'reason' => 'primary_file_not_found'
                ]));
                return;
            }

            $primaryTranslations = include $primaryPath;
            if (!is_array($primaryTranslations)) {
                Log::error("Translation job failed - invalid translation file format", array_merge($logContext, [
                    'primary_path' => $primaryPath,
                    'status' => 'failed',
                    'reason' => 'invalid_file_format'
                ]));
                return;
            }

            // Load existing secondary language file
            $secondaryTranslations = [];
            $fileExists = File::exists($secondaryPath);
            if ($fileExists) {
                $secondaryTranslations = include $secondaryPath;
                if (!is_array($secondaryTranslations)) {
                    $secondaryTranslations = [];
                    Log::warning("Existing secondary file has invalid format, will be recreated", array_merge($logContext, [
                        'secondary_path' => $secondaryPath
                    ]));
                }
            }

            // Prepare texts to translate (only missing ones)
            $textsToTranslate = $this->getMissingTranslations($primaryTranslations, $secondaryTranslations);

            if (empty($textsToTranslate)) {
                // All translations exist, just update progress
                $this->updateProgress();
                return;
            }

            // Call translation API with batch format (format 2: array of strings)
            $translations = $this->callTranslationApi($textsToTranslate);

            if ($translations && is_array($translations)) {
                // Merge new translations with existing ones
                $this->mergeTranslations($secondaryTranslations, $primaryTranslations, $translations, $textsToTranslate);

                // Ensure directory exists
                $directory = dirname($secondaryPath);
                if (!File::isDirectory($directory)) {
                    File::makeDirectory($directory, 0755, true);
                }

                // Write to file
                $this->writeTranslationsToFile($secondaryPath, $secondaryTranslations);

                // Verify file was created
                if (!File::exists($secondaryPath)) {
                    Log::error("Translation file creation failed - file does not exist after write", array_merge($logContext, [
                        'status' => 'failed',
                        'secondary_path' => $secondaryPath,
                        'reason' => 'file_not_created_after_write'
                    ]));
                    throw new \Exception("File was not created after write operation");
                }
            } else {
                // API failed - this is a failure, not a skip, since the file needs to be created
                $errorMessage = "Translation API call failed or returned invalid data";
                Log::error("Translation file creation failed - API error", array_merge($logContext, [
                    'status' => 'failed',
                    'reason' => 'translation_api_failed',
                    'secondary_path' => $secondaryPath,
                    'attempt' => $this->attempts(),
                    'max_attempts' => $this->tries
                ]));

                // Throw exception to trigger retry mechanism
                throw new \Exception($errorMessage);
            }

            $this->updateProgress();

        } catch (\Throwable $e) {
            Log::error("Translation job failed with exception", array_merge($logContext, [
                'status' => 'failed',
                'reason' => 'exception',
                'error_message' => $e->getMessage(),
                'error_class' => get_class($e),
                'attempt' => $this->attempts(),
                'max_attempts' => $this->tries,
                'secondary_path' => $secondaryPath ?? 'unknown'
            ]));

            throw $e;
        }
    }

    /**
     * Get missing translations
     *
     * @param array $primary
     * @param array $secondary
     * @return array
     */
    protected function getMissingTranslations(array $primary, array $secondary): array
    {
        $missing = [];

        foreach ($primary as $key => $value) {
            if (is_array($value)) {
                // Handle nested arrays recursively
                $nestedMissing = $this->getNestedMissing($value, $secondary[$key] ?? []);
                $missing = array_merge($missing, $nestedMissing);
            } else {
                // Check if translation exists and is not empty
                if (!isset($secondary[$key]) || empty($secondary[$key]) || $secondary[$key] === $key) {
                    $missing[] = $value;
                }
            }
        }

        return array_unique($missing);
    }

    /**
     * Get missing translations from nested arrays
     *
     * @param array $primary
     * @param array $secondary
     * @return array
     */
    protected function getNestedMissing(array $primary, array $secondary): array
    {
        $missing = [];

        foreach ($primary as $key => $value) {
            if (is_array($value)) {
                $nestedMissing = $this->getNestedMissing($value, $secondary[$key] ?? []);
                $missing = array_merge($missing, $nestedMissing);
            } else {
                if (!isset($secondary[$key]) || empty($secondary[$key]) || $secondary[$key] === $key) {
                    $missing[] = $value;
                }
            }
        }

        return $missing;
    }

    /**
     * Call translation API
     *
     * @param array $texts
     * @return array|null
     */
    protected function callTranslationApi(array $texts): ?array
    {
        $apiUrl = env('SECONDRY_LANG_TRANSLATION_FROM');
        $langCode = env('SECONDRY_LANG_CODE');

        if (!$apiUrl || !$langCode) {
            Log::warning("Translation API configuration missing", [
                'file' => $this->fileName,
                'api_url_set' => !empty($apiUrl),
                'lang_code_set' => !empty($langCode)
            ]);
            return null;
        }

        try {
            // Use format 2: array of strings
            $payload = [
                'text' => array_values($texts),
                'lang' => $langCode
            ];

            // Increase timeout for large translation batches
            $timeout = env('TRANSLATION_API_TIMEOUT', 120); // Default 2 minutes, configurable
            $response = Http::timeout((int)$timeout)
                ->post($apiUrl, $payload);

            if ($response->successful()) {
                $data = $response->json();

                if (isset($data['success']) && $data['success'] === true && isset($data['translation'])) {
                    $translation = $data['translation'];

                    // API returns object with original text as keys
                    // Example: { "Description": "記述", "Change In": "変化する" }
                    if (is_array($translation)) {
                        return $translation;
                    } else {
                        Log::error("Translation API returned invalid translation format", [
                            'file' => $this->fileName,
                            'translation_type' => gettype($translation)
                        ]);
                    }
                } else {
                    Log::error("Translation API response indicates failure", [
                        'file' => $this->fileName,
                        'response_success' => $data['success'] ?? 'not_set',
                        'has_translation' => isset($data['translation'])
                    ]);
                }
            } else {
                Log::error("Translation API request failed", [
                    'file' => $this->fileName,
                    'status_code' => $response->status(),
                    'response_body' => $response->body()
                ]);
            }
        } catch (\Throwable $e) {
            Log::error("Translation API call exception", [
                'file' => $this->fileName,
                'error_message' => $e->getMessage(),
                'error_trace' => $e->getTraceAsString()
            ]);
        }

        return null;
    }

    /**
     * Merge translations into secondary language array
     *
     * @param array &$secondary
     * @param array $primary
     * @param array $translations
     * @param array $textsToTranslate
     * @return void
     */
    protected function mergeTranslations(array &$secondary, array $primary, array $translations, array $textsToTranslate): void
    {
        foreach ($primary as $key => $value) {
            if (is_array($value)) {
                if (!isset($secondary[$key]) || !is_array($secondary[$key])) {
                    $secondary[$key] = [];
                }
                $this->mergeTranslations($secondary[$key], $value, $translations, $textsToTranslate);
            } else {
                // Check if this value needs translation
                if (in_array($value, $textsToTranslate)) {
                    // API returns translations with original text as key
                    if (isset($translations[$value])) {
                        $secondary[$key] = $translations[$value];
                    } else {
                        // If translation not found, keep original
                        $secondary[$key] = $value;
                    }
                } elseif (!isset($secondary[$key])) {
                    // Keep original if not in translation list
                    $secondary[$key] = $value;
                }
            }
        }
    }

    /**
     * Write translations to PHP file
     *
     * @param string $path
     * @param array $translations
     * @return void
     */
    protected function writeTranslationsToFile(string $path, array $translations): void
    {
        try {
            // Sort keys alphabetically
            ksort($translations, SORT_NATURAL | SORT_FLAG_CASE);

            $content = "<?php\n\nreturn [\n";
            $content .= $this->formatTranslations($translations, 1);
            $content .= "];\n";

            $bytesWritten = File::put($path, $content);

            if ($bytesWritten === false) {
                Log::error("Failed to write translation file", [
                    'file' => $this->fileName,
                    'path' => $path,
                    'reason' => 'file_put_failed'
                ]);
                throw new \Exception("Failed to write translation file to {$path}");
            }
        } catch (\Throwable $e) {
            Log::error("Exception while writing translation file", [
                'file' => $this->fileName,
                'path' => $path,
                'error_message' => $e->getMessage(),
                'error_trace' => $e->getTraceAsString()
            ]);
            throw $e;
        }
    }

    /**
     * Format translations array recursively
     *
     * @param array $translations
     * @param int $indent
     * @return string
     */
    protected function formatTranslations(array $translations, int $indent = 1): string
    {
        $content = '';
        $spaces = str_repeat('    ', $indent);

        foreach ($translations as $key => $value) {
            $escapedKey = $this->escapePhpString($key);

            if (is_array($value)) {
                $content .= "{$spaces}'{$escapedKey}' => [\n";
                $content .= $this->formatTranslations($value, $indent + 1);
                $content .= "{$spaces}],\n";
            } else {
                $escapedValue = $this->escapePhpString($value);
                $content .= "{$spaces}'{$escapedKey}' => '{$escapedValue}',\n";
            }
        }

        return $content;
    }

    /**
     * Escape string for PHP array syntax
     *
     * @param string $string
     * @return string
     */
    protected function escapePhpString(string $string): string
    {
        return str_replace(
            ['\\', "'", "\n", "\r", "\t"],
            ['\\\\', "\\'", "\\n", "\\r", "\\t"],
            $string
        );
    }

    /**
     * Update progress
     *
     * @return void
     */
    protected function updateProgress(): void
    {
        $progress = Cache::get('translation_import_progress', [
            'total' => 0,
            'processed' => 0,
            'current_file' => '',
            'status' => 'processing'
        ]);

        $progress['processed'] = ($progress['processed'] ?? 0) + 1;
        Cache::put('translation_import_progress', $progress, now()->addHours(2));
    }

    /**
     * Handle a job failure.
     *
     * @param \Throwable $exception
     * @return void
     */
    public function failed(\Throwable $exception): void
    {
        $secondaryPath = base_path("resources/lang/{$this->secondaryLang}/{$this->fileName}.php");
        
        Log::error("Translation job permanently failed after all retries", [
            'file' => $this->fileName,
            'primary_lang' => $this->primaryLang,
            'secondary_lang' => $this->secondaryLang,
            'status' => 'failed_permanently',
            'secondary_path' => $secondaryPath,
            'error_message' => $exception->getMessage(),
            'error_class' => get_class($exception),
            'total_attempts' => $this->attempts(),
            'file_exists' => File::exists($secondaryPath)
        ]);

        // Track failed files in cache for reporting
        $failedFiles = Cache::get('translation_import_failed_files', []);
        $failedFiles[] = [
            'file' => $this->fileName,
            'primary_lang' => $this->primaryLang,
            'secondary_lang' => $this->secondaryLang,
            'secondary_path' => $secondaryPath,
            'error' => $exception->getMessage(),
            'failed_at' => now()->toDateTimeString(),
            'attempts' => $this->attempts()
        ];
        Cache::put('translation_import_failed_files', $failedFiles, now()->addHours(24));
    }
}

