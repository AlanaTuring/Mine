const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = 'https://orrvmkjcjfctubwrqgxd.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9ycnZta2pjamZjdHVid3JxZ3hkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI3NzM4MjIsImV4cCI6MjA2ODM0OTgyMn0.0w8S0wYDrbYseQFngnNcmE8Sw0B4hfQ-qiIMRgZR8-g';

const supabase = createClient(supabaseUrl, supabaseKey);

async function testConnection() {
  console.log('Testing Supabase connection...');
  
  try {
    // Test 1: Check if we can connect
    console.log('1. Testing basic connection...');
    const { data, error } = await supabase.from('profiles').select('count').limit(1);
    
    if (error) {
      console.error('❌ Connection failed:', error);
      return;
    }
    
    console.log('✅ Connection successful');
    
    // Test 2: Check profiles table
    console.log('2. Testing profiles table...');
    const { data: profiles, error: profilesError } = await supabase
      .from('profiles')
      .select('*')
      .limit(5);
    
    if (profilesError) {
      console.error('❌ Profiles table error:', profilesError);
    } else {
      console.log('✅ Profiles table accessible');
      console.log('   Found', profiles?.length || 0, 'profiles');
    }
    
    // Test 3: Check posts table
    console.log('3. Testing posts table...');
    const { data: posts, error: postsError } = await supabase
      .from('posts')
      .select('*')
      .limit(5);
    
    if (postsError) {
      console.error('❌ Posts table error:', postsError);
    } else {
      console.log('✅ Posts table accessible');
      console.log('   Found', posts?.length || 0, 'posts');
    }
    
    // Test 4: Check if user is authenticated
    console.log('4. Testing authentication...');
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError) {
      console.error('❌ Auth error:', authError);
    } else if (user) {
      console.log('✅ User authenticated:', user.email);
    } else {
      console.log('ℹ️  No user authenticated (this is normal for public access)');
    }
    
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
}

testConnection(); 