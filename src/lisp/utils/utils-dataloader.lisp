(in-package :nnl2.utils)

;; NNL2

;; Filepath: nnl2/src/lisp/utils/utils-dataloader.lisp
;; File: utils-dataloader.lisp

;; Contains a dataloader main logic

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defstruct (nnl2-internal-dataloader 
             ;;; (:documentation "The internal structure of a neural network data loader")
			 (:constructor make-nnl2-internal-dataloader)
             (:conc-name nnl2-internal-dataloader-))	 
			 
  (samples nil
   ;;; :documentation "Input data for training"
   )
   
  (labels nil
   ;;; :documentation "Target values (for teacher-led training)"
   )
   
  (batch-size nnl2.system:*default-batch-size* 
   :type integer
   ;;; :documentation "Number of samples processed in one forward/backward pass"
   )
  
  (size 0 
   :type integer
   ;;; :documentation "Total number of samples rows (for internal logic)"
   )
   
  (num-batches 0 
   :type integer
   ;;; :documentation "Number of complete batches after dataset division"
   )
   
  (get-as :view 
   :type (member :view :copy)
   ;;; :documentation "Data access method. :view for memory efficiency, :copy for safety"
   )
	
  (drop-last nil 
   :type boolean
   ;;; :documentation "When t, discards the last batch if it is smaller than batch-size"
   )
	
  (shuffle nil 
   :type boolean
   ;;; :documentation "When t, randomizes sample order before creating batches"
   )
   
  (batch-order nil 
   :type (or null vector)
   ;;; :documentation "Vector of batch indices for shuffled access"
   ))

(defun dataloader (samples labels &key (batch-size nnl2.system:*default-batch-size*) (get-as :view) (drop-last nil) (shuffle nil))
  "Create a new dataloader for the given samples and labels
  
   Args:
       samples: Input data tensor for training
       labels: Target values tensor for supervised learning
       batch-size (&key) (default: *default-batch-size*): Number of samples per batch 
       get-as (&key) (default: :view): Data access method. :view for memory efficiency or :copy for safety 
       drop-last (&key) (default: nil): When t, discards the last batch if it's smaller than batch-size
       shuffle (&key) (default: nil): When t, randomizes batch order before each epoch
   
   Returns:
       A new nnl2-internal-dataloader structure configured with the specified parameters
   
   Raises:
       Error if samples and labels have different sizes along dimension 0
       Error if batch-size is not positive
       Error if dataset size is zero
	   Warning if shuffle is enabled but dataset size is smaller than batch size
	   Warning if drop-last is enabled but dataset size is smaller than batch size
	   Error if handler-case is failed"
	   
  (handler-case
      (let* ((samples-size (nnl2.hli.ad:nrows samples))
             (labels-size (nnl2.hli.ad:nrows labels))
			 
             (num-batches (if drop-last
                            (floor samples-size batch-size)
                            (ceiling samples-size batch-size)))
							
             (batch-order (when shuffle 
                            (shuffle-vector (coerce (loop for i from 0 below num-batches collect i) 'vector)))))
       
        (when (/= samples-size labels-size)
          (nnl2.log:error "Samples size (~A) does not match labels size (~A)" samples-size labels-size)
          (error "Samples and labels must have the same size along dimension 0"))
        
        (when (<= samples-size 0)
          (nnl2.log:error "Dataset size must be positive, got ~A" samples-size)
          (error "Dataset size must be positive"))
        
        (when (<= batch-size 0)
          (nnl2.log:error "Batch size must be positive, got ~A" batch-size)
          (error "Batch size must be positive"))
        
        (when (and shuffle (< samples-size batch-size))
          (nnl2.log:warning "Shuffle is enabled but dataset size (~A) is smaller than batch size (~A)" 
                           samples-size batch-size))
        
        (when (and drop-last (< samples-size batch-size))
          (nnl2.log:warning "Drop-last is enabled but dataset size (~A) is smaller than batch size (~A)" 
                           samples-size batch-size))
        
        (make-nnl2-internal-dataloader :samples samples
                                       :labels labels
                                       :batch-size batch-size
                                       :size samples-size
                                       :num-batches num-batches
                                       :get-as get-as
                                       :drop-last drop-last
                                       :shuffle shuffle
                                       :batch-order batch-order))
    (error (e)
      (nnl2.log:error "Failed to create dataloader: ~A" e)
      (signal e))))

(defun shuffle-vector (vector)
  "Shuffle the given vector in-place using the Fisher–Yates algorithm
  
   Args:
       vector: A simple vector whose elements will be shuffled in-place
   
   Returns:
       The same vector, but with its elements randomly shuffled
   
   Raises:
       Error if vector is not a valid array
       Error if the shuffle operation fails (wrapped and re-signaled by handler-case)"
	   
  (handler-case
      (loop for i from (1- (length vector)) downto 1
            do (let ((j (random (1+ i)))) (rotatef (aref vector i) (aref vector j)))
			finally (return vector))
			
    (error (e)
      (nnl2.log:error "Failed to shuffle vector: ~A" e)
      (signal e))))


(defun reset-shuffle (dataloader)
  "Reset the batch order for the dataloader if shuffling is enabled
  
   Args:
       dataloader: An nnl2-internal-dataloader structure whose batch order should be re-shuffled
   
   Returns:
       Nil, but updates the dataloader in-place by modifying its batch-order slot
       (only when shuffle is enabled and num-batches > 0)
   
   Raises:
       Error if dataloader is invalid
       Error if resetting the shuffle fails (wrapped and re-signaled by handler-case)"
	   
  (handler-case
      (when (nnl2-internal-dataloader-shuffle dataloader)
        (let ((num-batches (nnl2-internal-dataloader-num-batches dataloader)))
          (when (> num-batches 0)
            (setf (nnl2-internal-dataloader-batch-order dataloader)
                  (shuffle-vector (coerce (loop for i from 0 below num-batches collect i) 'vector))))))
				  
    (error (e)
      (nnl2.log:error "Failed to reset shuffle for dataloader: ~A" e)
      (signal e))))

(defun make-batch-tensor (tensor start length as)
  "Create a batch tensor as either a view or a copy
  
   Args:
       tensor: Input tensor from which a batch slice will be extracted
       start: Starting index along dimension 0
       length: Number of rows to include in the batch
       as: Either :view to return a lightweight narrowed view, or :copy to create an actual copied tensor
   
   Returns:
       A tensor representing the selected batch region
       If as = :view, returns a narrowed view without copying memory
       If as = :copy, returns a newly allocated tensor containing a copy of the batch slice
   
   Raises:
       Error if start is out of bounds
       Error if the batch range [start, start+length) exceeds tensor size
       Error if as is not one of :view or :copy
       Error if narrowing or copying fails (wrapped and re-signaled by handler-case)"
	   
  (handler-case
      (let ((tensor-size (nnl2.hli.ad:nrows tensor)))
        (when (>= start tensor-size)
          (nnl2.log:error "Start index ~A is out of bounds for tensor of size ~A" start tensor-size)
          (error "Start index out of bounds"))
        
        (when (> (+ start length) tensor-size)
          (nnl2.log:error "Batch range [~A, ~A) is out of bounds for tensor of size ~A" start (+ start length) tensor-size)
          (error "Batch range out of bounds"))
        
        (let ((narrowed (nnl2.hli.ad.utils:narrow tensor :dim 0 :start start :len length :track-graph nil)))
          (ecase as
            (:view narrowed)
            (:copy (nnl2.hli.ad:tlet ((copy (nnl2.hli.ad:copy narrowed)))
                     copy)))))
					 
    (error (e)
      (nnl2.log:error "Failed to create batch tensor: ~A" e)
      (signal e))))

(defmacro with-batch (dataloader (batch-x batch-y epoch) &body body)
  "Execute body with the batch tensors for the given epoch
  
   Args:
	   dataloader: An nnl2-internal-dataloader instance  
   
       (dataloader batch-x batch-y epoch): 
           batch-x: Symbol to bind the input batch tensor  
           batch-y: Symbol to bind the target batch tensor  
           epoch? Batch index for the current iteration (0-based)
		   
       body (&body): The code that will be executed with batch-x and batch-y bound
   
   Raises:
       Error if the dataloader is invalid
       Error if shuffle is enabled but batch-order is inconsistent  
       Error if batch extraction fails (e.g., make-batch-tensor error)
       Error if BODY signals an error (re-signaled after logging)
   
   Notes:
       Automatically resets shuffle if shuffle is enabled but batch-order is nil  
       Ensures no out-of-bounds batch extraction  
       Avoids executing body when the batch would be empty"
	   
  (let ((dataloader-var (gensym "DATALOADER-"))
        (epoch-var (gensym "EPOCH-"))
        (batch-size (gensym "BATCH-SIZE-"))
        (start-index (gensym "START-INDEX-"))
        (current-batch-size (gensym "CURRENT-BATCH-SIZE-"))
        (get-as (gensym "GET-AS-"))
        (real-epoch (gensym "REAL-EPOCH-")))
    
    `(let* ((,dataloader-var ,dataloader)
            (,epoch-var ,epoch)
			
            (,real-epoch
			  (if (nnl2-internal-dataloader-shuffle ,dataloader-var)
				(progn
				  (unless (nnl2-internal-dataloader-batch-order ,dataloader-var)
					(reset-shuffle ,dataloader-var))
       
				  (aref (nnl2-internal-dataloader-batch-order ,dataloader-var) ,epoch-var))
				  
				,epoch-var))
						   
            (,batch-size (nnl2-internal-dataloader-batch-size ,dataloader-var))
            (,start-index (* ,batch-size ,real-epoch))
			
            (,current-batch-size (min ,batch-size 
                                      (- (nnl2-internal-dataloader-size ,dataloader-var) 
                                         ,start-index)))
										 
            (,get-as (nnl2-internal-dataloader-get-as ,dataloader-var)))
       
       (when (and (> ,current-batch-size 0)
                  (< ,start-index (nnl2-internal-dataloader-size ,dataloader-var)))
				  
         (handler-case
             (nnl2.hli.ad:tlet 
               ((,batch-x (make-batch-tensor (nnl2-internal-dataloader-samples ,dataloader-var)
                                              ,start-index ,current-batch-size ,get-as))
											  
                (,batch-y (make-batch-tensor (nnl2-internal-dataloader-labels ,dataloader-var) 
                                              ,start-index ,current-batch-size ,get-as)))
											  
              ,@body)
			  
           (error (e)
             (nnl2.log:error "Error processing batch at epoch ~A (real epoch ~A): ~A" ,epoch-var ,real-epoch e)
             (signal e)))))))

(defmacro process (dataloader (batch-x batch-y &key (iterator (gensym "ITERATOR")) (shuffle :auto)) &body body)
  "Process all epochs of the dataloader, executing body for each batch
   
   Args:
       dataloader: An nnl2-internal-dataloader instance to iterate through
	   
       (batch-x batch-y &key iterator shuffle):
           batch-x: symbol to bind the input batch tensor
           batch-y: symbol to bind the label batch tensor
           iterator (&key) (default: fresh gensym): loop variable symbol 
           shuffle (&key) - shuffle behavior:
               :auto (default): use the dataloader's internal shuffle flag
               t: always reshuffle before processing
               nil: never shuffle
			   
       body (&body): Code to execute for each batch

   Raises:
       Error if the dataloader is invalid
       Error if batch-order or shuffling is inconsistent
       Error during batch extraction or body execution (re-signaled after logging)
       Warning if the dataloader contains zero batches
   
   Notes:
       If shuffle is :auto, the dataloader's internal shuffle flag determines behavior.
       If the dataloader has no batches, processing stops early with a warning.
       All batch tensor allocations happen inside nnl2.hli.ad:tlet regions
       provided by with-batch"
		 
  (let ((dataloader-var (gensym "DATALOADER-")))
    `(block process
	   (let ((,dataloader-var ,dataloader))
         (handler-case
             (progn
               (let ((num-batches (nnl2-internal-dataloader-num-batches ,dataloader-var)))
                 (when (<= num-batches 0)
                   (nnl2.log:warning "Dataloader has no batches to process")
                   (return-from process nil))

                 (let ((should-shuffle (cond ((eq ,shuffle t) t) ((eq ,shuffle nil) nil)  
											(t (nnl2-internal-dataloader-shuffle ,dataloader-var)))))
											
                   (when should-shuffle
                     (reset-shuffle ,dataloader-var)))
               
                 (dotimes (,iterator num-batches)
                   (with-batch ,dataloader-var (,batch-x ,batch-y ,iterator)
                     ,@body))))
         
           (error (e)
             (nnl2.log:error "Error during dataloader processing: ~A" e)
             (signal e)))))))
		   