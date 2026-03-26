<?php

namespace App\Http\Controllers;

use App\Jobs\TranslateLanguageFile;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Bus;
use Illuminate\Bus\Batch;
use Throwable;

class SecondaryLangController extends Controller
{
    /**
     * Start importing translations for secondary language
     *
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function importTranslations(Request $request)
    {
        try {
            $secondaryLang = env('SECONDY_LANG');
            $primaryLang = 'en';

            if (!$secondaryLang) {
                return response()->json([
                    'status' => false,
                    'message' => __('general.Secondary language is not configured')
                ]);
            }

            $primaryLangPath = base_path("resources/lang/{$primaryLang}");
            $secondaryLangPath = base_path("resources/lang/{$secondaryLang}");

            if (!File::exists($primaryLangPath)) {
                return response()->json([
                    'status' => false,
                    'message' => __('general.Primary language directory not found')
                ]);
            }

            // Get all PHP files from primary language directory
            $files = File::glob($primaryLangPath . '/*.php');

            if (empty($files)) {
                return response()->json([
                    'status' => false,
                    'message' => __('general.No translation files found')
                ]);
            }

            // Reset progress
            Cache::put('translation_import_progress', [
                'total' => count($files),
                'processed' => 0,
                'current_file' => '',
                'status' => 'processing'
            ], now()->addHours(2));

            // Create jobs for each file
            $jobs = [];
            foreach ($files as $file) {
                $fileName = basename($file, '.php');
                $jobs[] = new TranslateLanguageFile($fileName, $primaryLang, $secondaryLang);
            }

            // Dispatch batch
            $batch = Bus::batch($jobs)
                ->name('Import Translations')
                ->allowFailures()
                ->dispatch();

            // Store batch ID for tracking
            Cache::put('translation_batch_id', $batch->id, now()->addHours(2));

            return response()->json([
                'status' => true,
                'message' => __('general.Translation import started'),
                'batch_id' => $batch->id,
                'total_files' => count($files)
            ]);

        } catch (Throwable $e) {
            return response()->json([
                'status' => false,
                'message' => $e->getMessage()
            ]);
        }
    }

    /**
     * Get translation import progress
     *
     * @return \Illuminate\Http\JsonResponse
     */
    public function getProgress()
    {
        $progress = Cache::get('translation_import_progress', [
            'total' => 0,
            'processed' => 0,
            'current_file' => '',
            'status' => 'idle'
        ]);

        $batchId = Cache::get('translation_batch_id');

        if ($batchId) {
            $batch = Bus::findBatch($batchId);

            if ($batch) {
                if ($batch->finished()) {
                    $progress['status'] = 'completed';
                    $progress['processed'] = $progress['total'];

                    // Clear batch ID after completion
                    Cache::forget('translation_batch_id');
                } elseif ($batch->cancelled()) {
                    $progress['status'] = 'cancelled';
                } else {
                    $progress['status'] = 'processing';
                    // Update processed count based on batch progress
                    $progress['processed'] = $batch->processedJobs();
                }
            }
        }

        // Check if all files are processed (even if batch status hasn't updated yet)
        if ($progress['status'] === 'processing' &&
            $progress['total'] > 0 &&
            $progress['processed'] >= $progress['total']) {
            $progress['status'] = 'completed';
            // Clear batch ID
            Cache::forget('translation_batch_id');
        }

        return response()->json($progress);
    }

    /**
     * Clear translation import progress
     *
     * @return \Illuminate\Http\JsonResponse
     */
    public function clearProgress()
    {
        Cache::forget('translation_import_progress');
        Cache::forget('translation_batch_id');

        return response()->json([
            'status' => true,
            'message' => __('general.Progress cleared')
        ]);
    }
}

