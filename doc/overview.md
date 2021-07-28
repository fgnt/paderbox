# Paderbox: A collection of utilities for audio / speech processing

 - `pb.io`
   - `pb.io.load`, `pb.io.dump`: Loads/saves an arbitary file. See docsting for supported formats.
     - `unsafe` argument can enable unsafe backends like `pickle`
   - `pb.io.load_{json,yaml,csv,hdf5}`, `pb.io.dump_{json,yaml,csv,hdf5}`, `pb.io.loads_{json,yaml,csv,hdf5}`, `pb.io.dumps_{json,yaml,csv,hdf5}`:
     - Load or dump/save some data in a particular format. The `s` in `dumps` and `loads` follow python convention to obtain/yield string or bytes representation of an object.
   - `pb.io.recursive_load_audio`: Recursive load of audio files.
   - `pb.io.{symlink,update_hdf5,mkdir_p}`
   - `pb.io.data_dir`: Collection of paths, loaded from enviroment and with defaults for our department file system.
   - `paderbox.io.atomic`: Atomic file operations. See docstrings for more information.
 - `pb.transform`:
   - `pb.transform.{stft,istft,STFT}`: Functions and class (`STFT`) to calculate the stft and its inverse.
   - Other transformations we either don't use, or rarely use.
   - `pb.transform.resample_sox`: Resample with `sox` binary
 - `from paderbox.visualization import plot, axes_context`:
   - `plot.{line,scatter}`: Make a line or scatter plot
   - `plot.stft`: Plot the stft signal
   - `with axes_context(columns=...) as axes`:
     - Context manager to change visualization parameters (e.g. grid, colors, ...)
     - Helper to create a plotting grid. Use `ax=axes.new` or `ax=axes.last`.
 - `pb.array`:
   - `pb.array.interval`: Helper to have a memory efficient 1D boolian array that represents intervals as replacement for numpy.
   - `pb.array.pad_axis`: Add an `axis` argument to `np.pad`.
   - `pb.array.morph`: Deprecated in favour of `einops` (https://github.com/arogozhnikov/einops)
   - `pb.array.segment_axis`: Segment a signal. Use an implementation detail of numpy to do it without memory replications.
 - `pb.utils`:
   - `pb.utils.process_caller.run_process`:
     - Wrapper around `subprocess.run` for better exception messages and other defaults.
   - `pb.utils.process_caller.run_processes`:
     - Run multiple processes in parallel.
   - `with pb.utils.debug_utils.debug_on: ...`: Invoke `pdb` on exception.
   - `pb.utils.mapping.Dispatcher`: Dict like, but verbose exception message on `KeyError`. No relevant overhead to `dict`.
   - `pb.utils.nested.{flatten,deflatten,nested_op,FlatView,...}`:
     - Utilities to work with nested objects.
   - `pb.utils.pandas_utils.py_query`: Alternative to `pd.DataFrame.query` that supports all python code.
   - `pb.utils.pandas_utils.squeeze_df`: Remove "boring" colums in dataframe (i.e. each row has same value in that column)
   - `pb.utils.pandas_utils.display_df`: Combine `IPython.display.display` with `squeeze_df`.
   - `pb.utils.pretty.{pretty,pprint}`: Uses `IPython.lib.pretty.*`, but displays a summay of large numpy arrays instead of the actual values.
   - `pb.utils.profiling.lprun`: Lineprofiler decorator. ToDo: Make internal doku public.
   - `python -m paderbox.utils.strip_solutions`: CLI Helper to create a template notebook from solution notebook.
   - `pb.utils.timer.TimerDict`: Helper to get runtime of a codeblock.

# Standalone

 - [`lazy_dataset`](https://github.com/fgnt/lazy_dataset): Process large datasets as if it was an iterable.
   - Inpur pipeline with lazy loading, transformations and parallel loading.
   - Not limited to any NN framework.
 - [`dlp_mpi`](https://github.com/fgnt/dlp_mpi):
   - Parallisation with MPI based on `mpi4py`

# Special purpose packages

 - [`padertorch`](https://github.com/fgnt/padertorch)
   - A collection of common functionality to simplify the design, training and evaluation of machine learning models based on pytorch with an emphasis on speech processing.
 - [`pb_bss`](https://github.com/fgnt/pb_bss): Code related to blind source separation.
   - Metrics: `pb_bss.evaluation.{???}`
   - Beamforming: `pb_bss.extraction.{???}`
   - (Spatial) Mixture Models: `pb_bss.distribution.{???}`
 - [`nara_wpe`](https://github.com/fgnt/nara_wpe): Weighted Prediction Error
   - Dereverberation code: `nara_wpe.???`
 - [`sms_wsj`](https://github.com/fgnt/sms_wsj): SMS-WSJ: Spatialized Multi-Speaker Wall Street Journal database for multi-channel source separation and recognition
 - [`paderwasn`](https://github.com/fgnt/paderwasn):

# Example code

 - [`nn-gev`](https://github.com/fgnt/nn-gev)
 - [`pb_chime5`](https://github.com/fgnt/pb_chime5)
 - [`pb_sed`](https://github.com/fgnt/pb_sed)
 - [`upb_audio_tagging_2019`](https://github.com/fgnt/upb_audio_tagging_2019)
 - [`sins`](https://github.com/fgnt/sins)
