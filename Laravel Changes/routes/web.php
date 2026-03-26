// Secondary Language Translation Routes
    Route::controller(SecondaryLangController::class)->group(function () {
        Route::post('import_translations', 'importTranslations')->name('import_translations')->middleware('permission:View Settings');
        Route::get('translation_progress', 'getProgress')->name('translation_progress');
        Route::get('translation_progress/clear', 'clearProgress')->name('translation_progress_clear');
    });