"use client"

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getSupabaseClient } from "@/lib/supabaseClient";

interface DebugInfoProps {
  userId?: string;
}

export function DebugInfo({ userId }: DebugInfoProps) {
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [tableExists, setTableExists] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkDatabase();
  }, []);

  const checkDatabase = async () => {
    try {
      // Check current user
      const supabase = getSupabaseClient();
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      setCurrentUser(user?.id || null);

      if (userError) {
        setError(`User error: ${userError.message}`);
        return;
      }

      // Check if user_follows table exists
      const { data: tableCheck, error: tableError } = await supabase
        .from('user_follows')
        .select('count', { count: 'exact', head: true })
        .limit(1);

      if (tableError) {
        setTableExists(false);
        setError(`Table error: ${tableError.message}`);
      } else {
        setTableExists(true);
      }
    } catch (err) {
      setError(`General error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <Card className="border-orange-200 bg-orange-50">
      <CardHeader>
        <CardTitle className="text-orange-800">Debug Info</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        <div className="space-y-2">
          <div>
            <strong>Current User:</strong> {currentUser || 'Not logged in'}
          </div>
          <div>
            <strong>Target User:</strong> {userId || 'Not provided'}
          </div>
          <div>
            <strong>User_follows Table:</strong> {tableExists === null ? 'Checking...' : tableExists ? 'Exists' : 'Missing'}
          </div>
          {error && (
            <div className="text-red-600">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 