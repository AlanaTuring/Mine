// Test script to debug likes functionality
// Run this in browser console to test database connectivity

async function testLikesTable() {
  console.log('=== TESTING LIKES TABLE ===');
  
  try {
    // Import Supabase client (you'll need to run this in the browser console)
    const { createClient } = await import('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2');
    
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    );
    
    // Test 1: Check if likes table exists
    console.log('1. Checking if likes table exists...');
    const { data: tables, error: tableError } = await supabase
      .from('information_schema.tables')
      .select('table_name')
      .eq('table_schema', 'public')
      .eq('table_name', 'likes');
    
    console.log('Tables result:', tables, 'Error:', tableError);
    
    // Test 2: Check likes table structure
    console.log('2. Checking likes table structure...');
    const { data: columns, error: columnError } = await supabase
      .from('information_schema.columns')
      .select('column_name, data_type')
      .eq('table_schema', 'public')
      .eq('table_name', 'likes');
    
    console.log('Columns result:', columns, 'Error:', columnError);
    
    // Test 3: Check if we can insert a test like
    console.log('3. Testing like insertion...');
    const testLike = {
      post_id: 'test-post-id',
      user_id: 'test-user-id'
    };
    
    const { data: insertData, error: insertError } = await supabase
      .from('likes')
      .insert(testLike)
      .select();
    
    console.log('Insert test result:', insertData, 'Error:', insertError);
    
    // Test 4: Check RLS policies
    console.log('4. Checking RLS policies...');
    const { data: policies, error: policyError } = await supabase
      .from('pg_policies')
      .select('policyname, cmd')
      .eq('tablename', 'likes');
    
    console.log('Policies result:', policies, 'Error:', policyError);
    
  } catch (error) {
    console.error('Test failed:', error);
  }
}

// Run the test
testLikesTable(); 