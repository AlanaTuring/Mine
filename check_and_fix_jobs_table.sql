-- Check and Fix Jobs Table Issues
-- Run this in your Supabase SQL Editor

-- 1. Check if jobs table exists
SELECT 
    table_name,
    table_schema
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name = 'jobs';

-- 2. If jobs table exists, enable RLS and create policies
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'jobs') THEN
        -- Enable RLS
        ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;
        
        -- Create basic policies (adjust based on your actual jobs table structure)
        -- First, let's see what columns exist in the jobs table
        RAISE NOTICE 'Jobs table exists - enabling RLS';
        
        -- Create a basic view policy (adjust the column name based on your actual structure)
        EXECUTE 'CREATE POLICY "Users can view all jobs" ON public.jobs FOR SELECT USING (true)';
        
        -- If there's a user_id column, create user-specific policies
        IF EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = 'public' 
                AND table_name = 'jobs' 
                AND column_name = 'user_id'
        ) THEN
            EXECUTE 'CREATE POLICY "Users can create jobs" ON public.jobs FOR INSERT WITH CHECK (auth.uid() IS NOT NULL)';
            EXECUTE 'CREATE POLICY "Users can update their own jobs" ON public.jobs FOR UPDATE USING (auth.uid() = user_id)';
            EXECUTE 'CREATE POLICY "Users can delete their own jobs" ON public.jobs FOR DELETE USING (auth.uid() = user_id)';
        ELSE
            -- If no user_id column, create basic policies
            EXECUTE 'CREATE POLICY "Users can create jobs" ON public.jobs FOR INSERT WITH CHECK (auth.uid() IS NOT NULL)';
            EXECUTE 'CREATE POLICY "Users can update jobs" ON public.jobs FOR UPDATE USING (auth.uid() IS NOT NULL)';
            EXECUTE 'CREATE POLICY "Users can delete jobs" ON public.jobs FOR DELETE USING (auth.uid() IS NOT NULL)';
        END IF;
        
    ELSE
        RAISE NOTICE 'Jobs table does not exist - this is expected if you are not using jobs functionality';
    END IF;
END $$;

-- 3. Show current RLS status for all tables
SELECT 
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY tablename; 